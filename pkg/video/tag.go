package video

import (
	"bufio"
	"errors"
	"image"
	"image/color"
	"io"
	"log"
	"os"
	"os/exec"
	"path"
	"sort"
	"strconv"
	"strings"

	"github.com/chenBenjamin97/final-project/pkg/utils"
	"github.com/spf13/viper"
	"gocv.io/x/gocv"
)

const jointsRecognitionModelPath = "./openpose/graph_opt.pb"

var lightUniformTeamColor = color.RGBA{0, 255, 0, 0}
var darkUniformTeamColor = color.RGBA{255, 0, 0, 0}
var refreeTeamColor = color.RGBA{0, 0, 255, 0}

//Tag reads a video from given source, uses multiple machine learning models and uses openCV in order to plot above it's frames
//relevant data. The tagged (XVID (== MPEG-4 codec) format, '.avi' extension) will be saved in 'ready' directory from configuration file.
//srcVideoName should include file's extension ('.mp4', etc.)
func Tag(srcVideoName string) {
	srcVideoPath := path.Join(viper.GetString("directory.source"), srcVideoName)
	tmpVideoPath := path.Join(viper.GetString("directory.temp"), strings.Split(srcVideoName, ".")[0]+"."+"avi")
	outputVideoPath := path.Join(viper.GetString("directory.ready"), srcVideoName)

	cap, err := gocv.VideoCaptureFile(srcVideoPath)
	if err != nil {
		log.Printf("Tag: Error, Got '%v'", err)
		return
	}
	defer cap.Close()

	videoWriter, err := gocv.VideoWriterFile(tmpVideoPath, "XVID", cap.Get(gocv.VideoCaptureFPS), int(cap.Get(gocv.VideoCaptureFrameWidth)), int(cap.Get(gocv.VideoCaptureFrameHeight)), true)
	if err != nil {
		log.Printf("Tag: Error, Got '%v'", err)
		return
	}
	defer videoWriter.Close()
	defer os.Remove(tmpVideoPath) //remove '.avi' temp file at the end of this function

	framesObjectsStatsC := make(chan []*frameObjects)

	go RunTracker(srcVideoPath, framesObjectsStatsC)

	frameMat := gocv.NewMat()
	defer frameMat.Close()

	writtenFramesCounter := 0

mainLoop:
	for {
		select {
		case framesObjectsStats, ok := <-framesObjectsStatsC:
			if !ok { //sender closed chan
				break mainLoop
			} else {
				//data structure looks like: map[personID][timestamps][(x1,y1)..(x13,y13)]
				personsKeypointsTimestamps := make(map[int][][]image.Point)

				//will be used when we plot data on the frames, after we analyze them (instead of open 2 readers for this video file)
				recordedFrames := make([]gocv.Mat, 0)

				//data structure looks like: map[personID]personPrediction
				personPredictionMap := make(map[int]*personPrediction)

				idsToMarkAsRefs := matchingRefereesAndPersonsBboxes(framesObjectsStats)
				for _, id := range idsToMarkAsRefs {
					personPredictionMap[id] = &personPrediction{id: id}
					personPredictionMap[id].finalAction = utils.RefereeTeamID
					personPredictionMap[id].action1 = "Referee"
					personPredictionMap[id].confidence1 = "80.00%"
					personPredictionMap[id].teamID = utils.RefereeTeamID //flag to represent referee "team"
				}

				//a counter for each person ID how many times it's appearing as each team on given frames sequence.
				//will be used later to choose what color to draw it's bounding box
				//structure: map[personID][team 0 counter, team 1 counter]
				personIdsTeamPerFrameCounter := make(map[int][]int)

				//iterate over frames and collect joints for each person in each frame
				//len(framesObjectsStats) will be == utils.SequenceLength unless this chunck is shorter (end of video)
				for frameIndex := 0; frameIndex < len(framesObjectsStats); frameIndex++ {
					if cap.Read(&frameMat) { //finished to read all video's frames
						recordedFrames = append(recordedFrames, gocv.NewMat())
						frameMat.CopyTo(&recordedFrames[frameIndex]) //copy current frame to our buffer, will be used later for plotting

						width, height := frameMat.Cols(), frameMat.Rows()

						for personID, boundingBox := range framesObjectsStats[frameIndex].personsBoundingBoxes {
							if _, ok := personPredictionMap[personID]; !ok && boundingBox.InCourt { //only if did not marked it's class yet (== not referee) and this person in court
								fixBbox(boundingBox, height, width)
								roiMat := frameMat.Region(image.Rect(boundingBox.Xmin, boundingBox.Ymin, boundingBox.Xmax, boundingBox.Ymax))
								if keyPoints, err := FindJoints(roiMat); err != nil {
									log.Printf("Tag: Error, got '%v'", err)
									continue
								} else {
									if _, ok := personsKeypointsTimestamps[personID]; !ok { //allocate memory for this personID if needed
										personsKeypointsTimestamps[personID] = make([][]image.Point, 0)
									}

									personsKeypointsTimestamps[personID] = append(personsKeypointsTimestamps[personID], keyPoints)
								}
							}
						}

						for id, teamID := range getPersonsIdsTeams(&frameMat, framesObjectsStats[frameIndex].personsBoundingBoxes, idsToMarkAsRefs) {
							if _, ok := personIdsTeamPerFrameCounter[id]; !ok {
								personIdsTeamPerFrameCounter[id] = make([]int, 2)
								personIdsTeamPerFrameCounter[id][0] = 0
								personIdsTeamPerFrameCounter[id][1] = 0
							}

							if teamID == utils.DarkTeamID {
								personIdsTeamPerFrameCounter[id][0]++
							}

							if teamID == utils.LightTeamID {
								personIdsTeamPerFrameCounter[id][1]++
							}
						}
					} else { //no frame read, finished to iterate over file
						break
					}
				}

				for k := range personsKeypointsTimestamps {
					personPredictionMap[k] = &personPrediction{id: k}                                    //not exist yet, checked in loop before                                   //init a key in prediction's map for this new prediction
					if paddedSeq := utils.PadSequence(personsKeypointsTimestamps[k]); paddedSeq == nil { //too short data - do not tag it on video
						delete(personsKeypointsTimestamps, k) //in order to not run action recognition model for it later, invalid/ missing data sequence
						personPredictionMap[k].action1 = "Unknown"
						personPredictionMap[k].confidence1 = "0%"
						// personPredictionMap[k].color = color.RGBA{255, 255, 255, 0} //Default color
					} else {
						personsKeypointsTimestamps[k] = paddedSeq
					}
				}

				//set color for each person
				for k, counter := range personIdsTeamPerFrameCounter {
					if counter[0] > counter[1] { //darker shirts team - team 0
						personPredictionMap[k].teamID = utils.DarkTeamID
					} else { //lighter team - team 1
						personPredictionMap[k].teamID = utils.LightTeamID
					}
				}

				cmd := exec.Command("python3", "action_prediction.py", "--model", "joints_lstm_new_encoding.h5", "--timestamps", strconv.Itoa(utils.SequenceLength), "--persons", strconv.Itoa(len(personsKeypointsTimestamps)), "--keypoints", strconv.Itoa(utils.KeypointsNum))
				stdin, err := cmd.StdinPipe()
				if err != nil {
					log.Printf("Tag: Error getting python's standart input, Got '%v'", err)
					return
				}

				stdout, err := cmd.StdoutPipe()
				if err != nil {
					log.Printf("Tag: Error getting python's standart output, Got '%v'", err)
					return
				}

				defer stdin.Close()
				defer stdout.Close()

				for k, timestamps := range personsKeypointsTimestamps {
					for _, singleTimestamp := range timestamps {
						io.WriteString(stdin, strconv.Itoa(k)+";")
						for i, v := range singleTimestamp {
							if i == len(singleTimestamp)-1 { //write data to python in a format it will be convenient our python code to parse
								io.WriteString(stdin, strconv.Itoa(v.X)+","+strconv.Itoa(v.Y))
							} else {
								io.WriteString(stdin, strconv.Itoa(v.X)+","+strconv.Itoa(v.Y)+";")
							}
						}
						io.WriteString(stdin, "\n")
					}
				}

				if err := cmd.Start(); err != nil {
					log.Printf("Tag: Error executing python's code, Got '%v'", err)
					return
				}

				scanner := bufio.NewScanner(stdout)
				for scanner.Scan() { //python's expected output is one line for each person, each contains "person ID;predicted action;confidence"
					splittedLine := strings.Split(scanner.Text(), ";")
					personID, predictedAction1, confidence1, predictedAction2, confidence2, predictedAction3, confidence3 := splittedLine[0], splittedLine[1], splittedLine[2], splittedLine[3], splittedLine[4], splittedLine[5], splittedLine[6]
					idInt, _ := strconv.Atoi(personID)
					if _, ok := personPredictionMap[idInt]; !ok { //init if not exist yet
						personPredictionMap[idInt] = &personPrediction{id: idInt}
					}
					personPredictionMap[idInt].action1 = predictedAction1
					personPredictionMap[idInt].confidence1 = confidence1
					personPredictionMap[idInt].action2 = predictedAction2
					personPredictionMap[idInt].confidence2 = confidence2
					personPredictionMap[idInt].action3 = predictedAction3
					personPredictionMap[idInt].confidence3 = confidence3
				}

				if err := cmd.Wait(); err != nil {
					log.Printf("Tag: Error waiting python's standart output, Got '%v'", err)
				}

				//find for each frame what person IDs are around the ball
				idsAroundBallPerFrame := make([]map[int][]int, len(framesObjectsStats))
				for i, frameStats := range framesObjectsStats {
					for _, obj := range frameStats.customObjectBoundingBoxes {
						if obj.Class == utils.BallClass {
							ballBbox := image.Rect(obj.Xmin, obj.Ymin, obj.Xmax, obj.Ymax)
							idsAroundBallPerFrame[i] = findPersonsAroundBall(ballBbox, frameStats.personsBoundingBoxes)
						}
					}
				}

				attackingTeamID := getAttackingTeamID(idsAroundBallPerFrame, personPredictionMap)
				setActionsBasedOnAllStats(idsAroundBallPerFrame, personPredictionMap, attackingTeamID)

				for i, frame := range recordedFrames {
					writtenFramesCounter++

					for personID := range framesObjectsStats[i].personsBoundingBoxes {
						if framesObjectsStats[i].personsBoundingBoxes[personID].InCourt {
							if personPredictionMap[personID].teamID == utils.DarkTeamID {
								if err := plotPersonOnFrame(&frame, framesObjectsStats[i].personsBoundingBoxes[personID], *personPredictionMap[personID], darkUniformTeamColor); err != nil {
									log.Printf("Tag: Error tagging video file '%v': Could not plot person ID %v on frame number %v, got '%v'. Skipping.", srcVideoPath, personID, writtenFramesCounter, err)
								}
							}

							if personPredictionMap[personID].teamID == utils.LightTeamID {
								if err := plotPersonOnFrame(&frame, framesObjectsStats[i].personsBoundingBoxes[personID], *personPredictionMap[personID], lightUniformTeamColor); err != nil {
									log.Printf("Tag: Error tagging video file '%v': Could not plot person ID %v on frame number %v, got '%v'. Skipping.", srcVideoPath, personID, writtenFramesCounter, err)
								}
							}

							if personPredictionMap[personID].teamID == utils.RefereeTeamID {
								if err := plotPersonOnFrame(&frame, framesObjectsStats[i].personsBoundingBoxes[personID], *personPredictionMap[personID], refreeTeamColor); err != nil {
									log.Printf("Tag: Error tagging video file '%v': Could not plot person ID %v on frame number %v, got '%v'. Skipping.", srcVideoPath, personID, writtenFramesCounter, err)
								}
							}
						}
					}

					for _, obj := range framesObjectsStats[i].customObjectBoundingBoxes {
						if obj.Xmin == 0 && obj.Ymin == 0 && obj.Xmax == 0 && obj.Ymax == 0 { //skip, invalid
							continue
						}

						if obj.Class == utils.DontPlotFlag {
							continue
						}

						if obj.Class == utils.BallClass {
							boundingBoxRect := image.Rect(obj.Xmin, obj.Ymin, obj.Xmax, obj.Ymax)
							gocv.Rectangle(&frame, boundingBoxRect, color.RGBA{255, 128, 0, 0}, 3)
						}

						if obj.Class == utils.HoopClass {
							boundingBoxRect := image.Rect(obj.Xmin, obj.Ymin, obj.Xmax, obj.Ymax)
							gocv.Rectangle(&frame, boundingBoxRect, color.RGBA{255, 255, 102, 0}, 3)
						}

						//print class here if YOLOv4 not found a person bounding box which matching this bounding box
						//TODO: Validate it's working good
						if obj.Class == utils.RefereeClass {
							boundingBoxRect := image.Rect(obj.Xmin, obj.Ymin, obj.Xmax, obj.Ymax)
							plotReferee(&frame, boundingBoxRect, refreeTeamColor)
							// gocv.Rectangle(&frame, boundingBoxRect, color.RGBA{0, 0, 255, 0}, 3)
						}
					}

					videoWriter.Write(frame)
				}
			}
		}
	}

	//Convert to from 'avi' to 'mp4'. example:ffmpeg -i testBasketball.avi testBasketball.mp4
	cmd := exec.Command("ffmpeg", "-i", tmpVideoPath, outputVideoPath)
	if err := cmd.Run(); err != nil {
		log.Printf("Tag: Error from ffmpeg, got '%v'", err)
	}
}

//FindJoints gets an roi of a frame and returns wanted joints points slice.
//In case no joints are found, a slice full of zeroes points will be returned, should ignore it in the action recognition model
func FindJoints(roi gocv.Mat) ([]image.Point, error) {
	net := gocv.ReadNetFromTensorflow(jointsRecognitionModelPath)
	if net.Empty() {
		return nil, errors.New("FindJoints: Could not load model")
	}
	defer net.Close()

	net.SetInput(gocv.BlobFromImage(roi, 1, image.Point{X: roi.Cols(), Y: roi.Rows()}, gocv.NewScalar(127.5, 127.5, 127.5, 127.5), true, false), "")
	prob := net.Forward("")
	s := prob.Size()
	nparts, h, w := s[1], s[2], s[3]

	nparts = utils.KeypointsNum               //we care only about utils.KeypointsNum keypoints (index 1-13)
	xAvg, yAvg, foundJointsCounter := 0, 0, 0 //will be used in order to handle needed joints that not found

	// find the most likely match for each part
	pts := make([]image.Point, utils.KeypointsNum) //force exactly utils.KeypointsNum keypoints will be returned
	for i := 0; i < nparts; i++ {
		pts[i] = image.Pt(-1, -1)
		heatmap, _ := prob.FromPtr(h, w, gocv.MatTypeCV32F, 0, i+1) //we want the next index from the prob mat, changed source code a bit

		_, conf, _, pt := gocv.MinMaxLoc(heatmap)
		if conf > 0.1 {
			pts[i] = pt
			xAvg += pt.X
			yAvg += pt.Y
			foundJointsCounter++
		}
		heatmap.Close()
	}

	avgPoint := image.Pt(0, 0) //default value
	if foundJointsCounter != 0 {
		avgPoint.X = xAvg / foundJointsCounter
		avgPoint.Y = yAvg / foundJointsCounter
	}

	//calculate average and set it where the point is not found
	for i, pt := range pts {
		if pt == image.Pt(-1, -1) || pt == image.Pt(0, 0) {
			pts[i] = avgPoint
		}
	}

	return pts, nil
}

//correspondingRects returns true in case of overlap of more than 'minCorrespondingRatio' on both axises of one of given rects
func correspondingRects(r1, r2 image.Rectangle) bool {
	minCorrespondingRatio := 0.75 //in case of overlap of more than 'minCorrespondingRatio' on both axises of one of given rects - return true

	if r1.In(r2) || r2.In(r1) { //one of given rects contains the other one
		return true
	}

	if intersectRect := r1.Intersect(r2); intersectRect.Empty() {
		return false //given rects are not overlaps at all
	} else {
		if float64(intersectRect.Dx()) >= minCorrespondingRatio*float64(r1.Dx()) && float64(intersectRect.Dy()) >= minCorrespondingRatio*float64(r1.Dy()) || (float64(intersectRect.Dx()) >= minCorrespondingRatio*float64(r2.Dx()) && float64(intersectRect.Dy()) >= minCorrespondingRatio*float64(r2.Dy())) {
			return true
		}
	}

	return false
}

//matchingRefereesAndPersonsBboxes returns a list of personsID which match refrees bounding box at least in 75% of person's total appearances in given frames.
//In addition, in each frame it marks a referee bounding box not to be plotted in case there is another bounding box which matching it
func matchingRefereesAndPersonsBboxes(frames []*frameObjects) []int {
	//slice built in this form: [#frame][persinID..]
	personIDsRefsFrames := make([][]int, 16)

	//count total appearances of each person in given frames
	totalAppearances := make(map[int]float32)
	for _, frame := range frames {
		for k := range frame.personsBoundingBoxes {
			if _, ok := totalAppearances[k]; !ok {
				totalAppearances[k] = 1
			} else {
				totalAppearances[k]++
			}
		}
	}

	//check for each frame if there is a person bounding box and referee bounding box which matches
	for i, frameStats := range frames {
		personIDsRefsFrames[i] = make([]int, 0)
		for i, obj := range frameStats.customObjectBoundingBoxes {
			if obj.Class == utils.RefereeClass {
				refRect := image.Rect(obj.Xmin, obj.Ymin, obj.Xmax, obj.Ymax)
				for k := range frameStats.personsBoundingBoxes {
					personRect := image.Rect(frameStats.personsBoundingBoxes[k].Xmin, frameStats.personsBoundingBoxes[k].Ymin, frameStats.personsBoundingBoxes[k].Xmax, frameStats.personsBoundingBoxes[k].Ymax)
					if correspondingRects(refRect, personRect) {
						personIDsRefsFrames[i] = append(personIDsRefsFrames[i], k)
						frameStats.customObjectBoundingBoxes[i].Class = utils.DontPlotFlag
					}
				}
			}
		}
	}

	//count how many matches each person id had over given frames
	matchesCounter := make(map[int]float32)
	for _, ids := range personIDsRefsFrames {
		for _, id := range ids {
			if _, ok := matchesCounter[id]; !ok {
				matchesCounter[id] = 1
			} else {
				matchesCounter[id]++
			}
		}
	}

	minMatchRatio := float32(0.75)

	res := make([]int, 0)
	for k, v := range matchesCounter {
		if v >= totalAppearances[k]*minMatchRatio {
			res = append(res, k)
		}
	}

	return res
}

//getPersonsIdsTeams returns a map of person ID's which their bounding box is marked as "in court", and based on their shirt's color it returns a map of: map[ID]teamID (teamID = 0/1)
//You can give a slice of IDs that will be ignored and not calculated although it has bounding box (relevant to make the median more accurate)
func getPersonsIdsTeams(frame *gocv.Mat, bboxes map[int]*personBoundingBox, idsToIgnore []int) map[int]int {
	if len(bboxes) == 0 {
		return nil
	}

	grayFrame := gocv.NewMat()
	defer grayFrame.Close()

	gocv.CvtColor(*frame, &grayFrame, gocv.ColorBGRToGray)

	avgValues := make([]float64, 0)
	avgValuesMap := make(map[int]float64)
	inCourtCounter := 0

mainLoop:
	for id, bbox := range bboxes {
		for _, ignoreID := range idsToIgnore {
			if ignoreID == id {
				continue mainLoop
			}
		}

		if bbox.InCourt {
			inCourtCounter++
			boxHeight := bbox.Ymax - bbox.Ymin
			boxWidth := bbox.Xmax - bbox.Xmin
			roiRect := image.Rect(bbox.Xmin+boxWidth/3, bbox.Ymin+boxHeight/3, bbox.Xmax-boxWidth/3, bbox.Ymax-boxHeight/3) //middle third of the bounding box, trying to catch uniform only
			roiGrayFrame := grayFrame.Region(roiRect)
			avg := roiGrayFrame.Mean()
			avgValues = append(avgValues, avg.Val1)
			avgValuesMap[id] = avg.Val1
		}
	}

	sort.Float64s(avgValues)
	median := avgValues[inCourtCounter/2]

	teamsMap := make(map[int]int)

	for id, avg := range avgValuesMap {
		if avg < median {
			teamsMap[id] = utils.DarkTeamID
		} else {
			teamsMap[id] = utils.LightTeamID
		}
	}

	return teamsMap
}

//findPersonsAroundBall returns a map with two keys: key 0 contains list of person IDs which contains in it the ball bounding box, key 1 contains a list of person IDs which their bounding boxes are intersecting
//with ball bounding box
func findPersonsAroundBall(ballBbox image.Rectangle, personBboxes map[int]*personBoundingBox) map[int][]int {
	resultMap := make(map[int][]int)
	resultMap[0] = make([]int, 0) //list of person IDs which the ball bounding box is in their bounding box
	resultMap[1] = make([]int, 0) //list of persons IDs which at least part of the ball bounding box is in their bounding box

	for id, bbox := range personBboxes {
		personBboxRect := image.Rect(bbox.Xmin, bbox.Ymin, bbox.Xmax, bbox.Ymax)
		if ballBbox.In(personBboxRect) {
			resultMap[0] = append(resultMap[0], id)
			continue
		} else {
			if intersectRect := personBboxRect.Intersect(ballBbox); !intersectRect.Empty() {
				resultMap[1] = append(resultMap[1], id)
			}
		}
	}

	return resultMap
}

//getAttackingTeamID returns team ID which more of it's player where around the ball in given frame sequence
func getAttackingTeamID(AroundBallPerFrameMap []map[int][]int, personsPredictionMap map[int]*personPrediction) int {
	counterTeamDark, counterTeamLight := 0, 0

	for _, m := range AroundBallPerFrameMap {
		if len(m[0]) > 0 {
			if personsPredictionMap[m[0][0]].teamID == utils.DarkTeamID { //if there is a bounding box which contains the ball - "reward" it's counter with 2 points and do not skip this frame
				counterTeamDark += 2
			} else if personsPredictionMap[m[0][0]].teamID == utils.LightTeamID {
				counterTeamLight += 2
			}
		} else {
			if len(m[1]) > 0 { //there might be a situation where both are empty => no one around the ball in this frame
				for _, id := range m[1] {
					if _, ok := personsPredictionMap[id]; ok { //make sure we have predicted this id
						if personsPredictionMap[id].teamID == utils.DarkTeamID {
							counterTeamDark += 1
						} else if personsPredictionMap[id].teamID == utils.LightTeamID {
							counterTeamLight += 1
						}
					}
				}
			}
		}
	}

	if counterTeamDark > counterTeamLight {
		return utils.DarkTeamID
	} else {
		return utils.LightTeamID
	}
}

//setActionsBasedOnAllStats sets 'finalAction' value for each person in personPredictionMap based on ball's location and attacking team ID
func setActionsBasedOnAllStats(idsAroundBallPerFrame []map[int][]int, personPredictionMap map[int]*personPrediction, attackingTeamID int) {
	//check who is controlling the ball (if there is any)
	idsAroundBallTotalCountMap := make(map[int]bool)
	for _, frame := range idsAroundBallPerFrame {
		for _, ids := range frame {
			for _, id := range ids {
				idsAroundBallTotalCountMap[id] = true
			}
		}
	}

	for id := range idsAroundBallTotalCountMap {
		if _, ok := personPredictionMap[id]; ok {
			if personPredictionMap[id].teamID == attackingTeamID {
				if utils.InSlice(personPredictionMap[id].action1, utils.AttackActionsWithBall) {
					personPredictionMap[id].finalAction = 1
					continue
				} else if utils.InSlice(personPredictionMap[id].action2, utils.AttackActionsWithBall) {
					personPredictionMap[id].finalAction = 2
					continue
				} else if utils.InSlice(personPredictionMap[id].action3, utils.AttackActionsWithBall) {
					personPredictionMap[id].finalAction = 3
					continue
				} else {
					personPredictionMap[id].finalAction = utils.DefaultActionNum
					continue
				}
			}
		}
	}

	//if arrived here - this person is not having the ball or stands around it
	for id, personPredict := range personPredictionMap {
		if personPredict.teamID == attackingTeamID {
			if _, ok := idsAroundBallTotalCountMap[id]; ok {
				continue //we have took care of this person earlier, he is attacking team player around the ball
			}

			if !utils.InSlice(personPredict.action1, utils.DefenseOnlyAction) && !utils.InSlice(personPredict.action1, utils.AttackActionsWithBall) {
				//set this as major action + continue
				personPredict.finalAction = 1
				continue
			} else if !utils.InSlice(personPredict.action2, utils.DefenseOnlyAction) && !utils.InSlice(personPredict.action1, utils.AttackActionsWithBall) {
				personPredict.finalAction = 2
				continue
			} else if !utils.InSlice(personPredict.action3, utils.DefenseOnlyAction) && !utils.InSlice(personPredict.action1, utils.AttackActionsWithBall) {
				personPredict.finalAction = 3
				continue
			} else {
				personPredict.finalAction = utils.DefaultActionNum
				continue
			}
		} else { //defense player
			if !utils.InSlice(personPredict.action1, utils.AttackOnlyActions) {
				//set this as major action + continue
				personPredict.finalAction = 1
				continue
			} else if !utils.InSlice(personPredict.action2, utils.AttackOnlyActions) {
				personPredict.finalAction = 2
				continue
			} else if !utils.InSlice(personPredict.action3, utils.AttackOnlyActions) {
				personPredict.finalAction = 3
				continue
			} else {
				personPredict.finalAction = utils.DefaultActionNum
				continue
			}
		}
	}
}

//fixBbox fixes bounding boxes values in case they are out of frame's range
func fixBbox(bbox *personBoundingBox, frameHeight, frameWidth int) {
	if bbox.Xmin < 0 {
		bbox.Xmin = 0
	} else if bbox.Xmin > frameWidth {
		bbox.Xmin = frameWidth
	}

	if bbox.Ymin < 0 {
		bbox.Ymin = 0
	} else if bbox.Ymin > frameHeight {
		bbox.Ymin = frameHeight
	}

	if bbox.Xmax < 0 {
		bbox.Xmax = 0
	} else if bbox.Xmax > frameWidth {
		bbox.Xmax = frameWidth
	}

	if bbox.Ymax < 0 {
		bbox.Ymax = 0
	} else if bbox.Ymax > frameHeight {
		bbox.Ymax = frameHeight
	}
}
