package utils

import (
	"fmt"
	"io/ioutil"
)

//InSlice returns true if given string appears in given slice
func InSlice(lookingFor string, slice []string) bool {
	for _, s := range slice {
		if s == lookingFor {
			return true
		}
	}

	return false
}

//ListDir returns a list of files/ directories in given path
func ListDir(path string) ([]string, error) {
	names := make([]string, 0)
	if files, err := ioutil.ReadDir(path); err != nil {
		return nil, fmt.Errorf("ListDir: Error, got '%v'", err)
	} else {
		for _, f := range files {
			names = append(names, f.Name())
		}
	}

	return names, nil
}
