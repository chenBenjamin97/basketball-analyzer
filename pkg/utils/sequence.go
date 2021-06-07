package utils

import "image"

//PadSequence makes sure given sequence has the wanted length. In case it's shorter, it's padding it.
//In case given sequence is shorter than half of wanted length - it returns nil (invalid data)
func PadSequence(seq [][]image.Point) [][]image.Point {
	if len(seq) == SequenceLength {
		return seq
	}

	if len(seq) > SequenceLength {
		return seq[:SequenceLength]
	}

	if len(seq) < SequenceLength*0.5 {
		return nil
	}

	originalLength := len(seq)
	for i := originalLength; i < SequenceLength; i++ {
		seq = append(seq, seq[len(seq)-1])
	}

	return seq
}
