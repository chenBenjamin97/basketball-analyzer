package video

import (
	"bufio"
	"encoding/json"
	"log"
	"os/exec"
	"strings"

	"github.com/chenBenjamin97/final-project/pkg/utils"
	"github.com/spf13/viper"
)

//RunTracker executes python code that uses YOLOv4 based on COCO dataset and uses DEEP-SORT in order to detect persons in each frame
//and track them between frames. This function listens to python's standard output, save it in data structre and each 16 frames sends the data
//through a chan to another function to handle it (in order to save time, process the data while recognizing persons in frames).
//Because this function is the only one who writes it given chan, it will close it brfore it's finishing.
func RunTracker(videoPath string, framesStatsC chan<- []*frameObjects) {
	cmd := exec.Command("python3", viper.GetString("directory.yolov4-deepsort"), "--video", videoPath)

	defer func(framesStatsC chan<- []*frameObjects) {
		close(framesStatsC)
	}(framesStatsC)

	//map structure: []personBoundingBox
	framesObjectSlice := make([]*frameObjects, 0)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Printf("RunTracker: Error, got '%v'", err)
		return
	}
	defer stdout.Close()

	if err := cmd.Start(); err != nil {
		log.Printf("RunTracker: Error, got '%v'", err)
		return
	}

	scanner := bufio.NewScanner(stdout)

	//runs in iterations of 16 (each 16 frames are one action)
	framesCounter := 0

	for scanner.Scan() {
		if strings.Contains(scanner.Text(), "Frame #:") {
			framesCounter++

			if framesCounter == utils.SequenceLength+1 { //means we have scanned 16 frames
				framesStatsC <- framesObjectSlice //pass data to other function to handle

				//allocate new slice in order to clear the stats in current function but not delete older valus passed to the chan above
				framesObjectSlice = make([]*frameObjects, 1)
				framesObjectSlice[0] = NewframeObjects(1)
				framesCounter = 1 //reset counter to first frame
			} else { //allocate new frame in frames stats map
				framesObjectSlice = append(framesObjectSlice, NewframeObjects(framesCounter))
			}

			continue
		}

		if scanner.Text() == "EOF" { //finished to read all frames - send left data to other function and close this goroutine
			framesStatsC <- framesObjectSlice
			return
		}

		if strings.Contains(scanner.Text(), "FPS: ") { //this is a log print, skip it
			continue
		}

		if strings.Contains(scanner.Text(), "{\"ID\":") { //it's printing detected person data
			p := personBoundingBox{}
			if err := json.Unmarshal(scanner.Bytes(), &p); err == nil {
				framesObjectSlice[framesCounter-1].personsBoundingBoxes[p.ID] = &p
			} else {
				log.Printf("RunTracker: Error, got '%v'", err)
			}
			continue
		}

		if strings.Contains(scanner.Text(), "{\"Class\":") {
			obj := customObjectBoundingBox{}
			if err := json.Unmarshal(scanner.Bytes(), &obj); err == nil {
				framesObjectSlice[framesCounter-1].customObjectBoundingBoxes = append(framesObjectSlice[framesCounter-1].customObjectBoundingBoxes, &obj)
			} else {
				log.Printf("RunTracker: Error, got '%v'", err)
			}
			continue
		}
	}

	if err := cmd.Wait(); err != nil {
		log.Printf("RunTracker: Error waiting python's process, Got '%v'", err)
		return
	}
}
