package video

type personBoundingBox struct {
	ID      int
	Xmin    int
	Ymin    int
	Xmax    int
	Ymax    int
	InCourt bool
}

type personPrediction struct {
	id          int
	action1     string //action which got the highest confidence
	action2     string //action which got the second highest confidence
	action3     string //action which got the third highest confidence
	confidence1 string
	confidence2 string
	confidence3 string
	teamID      int
	finalAction int //chosen action after checking for teamID and ball position
}

type customObjectBoundingBox struct {
	Class      int
	Confidence float32
	Xmin       int
	Ymin       int
	Xmax       int
	Ymax       int
}

type frameObjects struct {
	frameNumber               int
	personsBoundingBoxes      map[int]*personBoundingBox
	customObjectBoundingBoxes []*customObjectBoundingBox
}

func NewframeObjects(frameNum int) *frameObjects {
	x := frameObjects{}
	x.frameNumber = frameNum
	x.personsBoundingBoxes = make(map[int]*personBoundingBox)
	x.customObjectBoundingBoxes = make([]*customObjectBoundingBox, 0)
	return &x
}
