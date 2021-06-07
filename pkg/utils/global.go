package utils

//SequenceLength is the length needed for one action recognition
const SequenceLength = 16

//KeypointsNum is the number of wanted keypoints from each timestamp
const KeypointsNum = 13

//Ballclass is the enum represents an object detected as a ball
const BallClass = 0

//Hoopclass is the enum represents an object detected as a hoop
const HoopClass = 1

//Refereeclass is the enum represents an object detected as a referee
const RefereeClass = 2

//DontPlotFlag is a flag that marks we do not want to plot it's object bounding box on certain frame
const DontPlotFlag = -1

//DarkTeamID is an enum to represent the team with darker uniforms
const DarkTeamID = 10

//LightTeamID is an enum to represent the team with lighter uniforms
const LightTeamID = 20

//RefereeTeamID is an enum represents a boundong box of a referee
const RefereeTeamID = 30

//AttackOnlyActions is a list of actions only players belong to attacking team can have
var AttackOnlyActions = []string{"Pass", "Dribble", "Shoot", "Ball In Hand", "Pick"}

//AttackActionsWithBall is a list of actions only players belong to attacking team and has the ball can have
var AttackActionsWithBall = []string{"Dribble", "Shoot", "Ball In Hand"}

//DefenseOnlyActions is a list of actions only players belong to defense team can have
var DefenseOnlyAction = []string{"Defense", "Block"}

//DefaultAction is the default action for a person in case model's predicted action for it does not fit it's team
const DefaultActionName = "Walk"

//DefaultAction is the default action to set in 'finalAction' for a person in case model's predicted action for it does not fit it's team
const DefaultActionNum = 4

//DefaultActionConfidence is the default confidence for a person in case model's predicted action for it does not fit it's team
const DefaultActionConfidence = "50.00%"
