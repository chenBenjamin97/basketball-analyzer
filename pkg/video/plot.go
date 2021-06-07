package video

import (
	"errors"
	"fmt"
	"image"
	"image/color"

	"github.com/chenBenjamin97/final-project/pkg/utils"
	"gocv.io/x/gocv"
)

//plotPersonOnFrame plots given bounding box and writes above it given stats
func plotPersonOnFrame(frame *gocv.Mat, box *personBoundingBox, pred personPrediction, plotColor color.RGBA) error {
	//this is a situation when our tracker could not find bounding box for this object in this frame but we set it to {(0,0,),(0,0)} in order to complete the sequence for LSTM model.
	//should not be plotted on frame
	if box.Xmin == 0 && box.Ymin == 0 && box.Xmax == 0 && box.Ymax == 0 {
		return nil
	}

	if box.ID != pred.id {
		return errors.New("PlotPersonOnFrame: Mismatched ID's between given personBoundingBox and personPrediction objects")
	}

	var actionToPrint string
	var confidenceToPrint string
	switch pred.finalAction {
	case 1:
		actionToPrint = pred.action1
		confidenceToPrint = pred.confidence1
		break
	case 2:
		actionToPrint = pred.action2
		confidenceToPrint = pred.confidence2
		break
	case 3:
		actionToPrint = pred.action3
		confidenceToPrint = pred.confidence3
		break
	case utils.RefereeTeamID:
		actionToPrint = pred.action1         //it will be "Referee"
		confidenceToPrint = pred.confidence1 //it will be default for "Referee" - 80.00%
		break
	default:
		actionToPrint = utils.DefaultActionName
		confidenceToPrint = utils.DefaultActionConfidence
		break
	}

	boundingBoxRect := image.Rect(box.Xmin, box.Ymin, box.Xmax, box.Ymax)
	gocv.Rectangle(frame, boundingBoxRect, plotColor, 3)

	textToPutFirstLine := fmt.Sprintf("ID: %d", pred.id)
	textToPutSecondLine := fmt.Sprintf("%s: %s", actionToPrint, confidenceToPrint)
	startPointFirstLine := image.Pt(boundingBoxRect.Min.X, boundingBoxRect.Min.Y-20)
	startPointSecondLine := image.Pt(boundingBoxRect.Min.X, boundingBoxRect.Min.Y-5)

	var textBackgroundRect image.Rectangle
	if actionToPrint == "Ball In Hand" {
		textBackgroundRect = image.Rect(startPointFirstLine.X, startPointFirstLine.Y-15, startPointFirstLine.X+160, startPointFirstLine.Y+20) //thickness -1 == filled rectangle
	} else if actionToPrint == "Referee" || actionToPrint == "Defense" {
		textBackgroundRect = image.Rect(startPointFirstLine.X, startPointFirstLine.Y-15, startPointFirstLine.X+137, startPointFirstLine.Y+20) //thickness -1 == filled rectangle
	} else { //default
		textBackgroundRect = image.Rect(startPointFirstLine.X, startPointFirstLine.Y-15, startPointFirstLine.X+125, startPointFirstLine.Y+20) //thickness -1 == filled rectangle
	}

	whiteRGB := color.RGBA{255, 255, 255, 0}
	gocv.Rectangle(frame, textBackgroundRect, plotColor, -1)
	gocv.PutText(frame, textToPutFirstLine, startPointFirstLine, gocv.FontHersheyPlain, 1, whiteRGB, 2)
	gocv.PutText(frame, textToPutSecondLine, startPointSecondLine, gocv.FontHersheyPlain, 1, whiteRGB, 2)

	return nil
}

//plotReferee plot's Referee that do not have an ID (YOLOv4 did not found a person bounding box which matching this bounding box)
func plotReferee(frame *gocv.Mat, bbox image.Rectangle, plotColor color.RGBA) {
	whiteRGB := color.RGBA{255, 255, 255, 0}

	startPointText := image.Pt(bbox.Min.X, bbox.Min.Y)
	textBackgroundRect := image.Rect(startPointText.X, startPointText.Y, bbox.Max.X, startPointText.Y-25)

	gocv.Rectangle(frame, bbox, plotColor, 3)
	gocv.Rectangle(frame, textBackgroundRect, plotColor, -1) //thickness -1 == filled rectangle
	gocv.PutText(frame, "Referee", startPointText, gocv.FontHersheyPlain, 1, whiteRGB, 2)

}
