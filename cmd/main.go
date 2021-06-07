package main

import (
	"log"
	"os"

	"github.com/chenBenjamin97/final-project/pkg/api"
	"github.com/spf13/viper"
)

func main() {
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)

	viper.AddConfigPath(".")
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	if err := viper.ReadInConfig(); err != nil {
		log.Fatalf("Error: Could not read config file, got '%v'", err)
	}

	//first - create project's data root dir
	if _, err := os.Stat(viper.GetString("directory.root")); err != nil {
		if os.IsNotExist(err) {
			if os.Mkdir(viper.GetString("directory.root"), 0766) != nil {
				log.Printf("Error Creating '%s' directory, got '%v'", viper.GetString("directory.root"), err)
			}
		}
	}

	//create missing directories from config file
	for _, dir := range viper.GetStringMap("directory") {
		if _, err := os.Stat(dir.(string)); err != nil {
			if os.IsNotExist(err) {
				if os.Mkdir(dir.(string), 0766) != nil {
					log.Printf("Error Creating '%s' directory, got '%v'", dir.(string), err)
				}
			}
		}
	}

	if viper.GetString("video.prod_format") == "" || viper.GetString("directory.yolov4-deepsort") == "" || viper.GetString("frontend.static-files-path") == "" {
		log.Fatalf("Error: Missing critical configurations")
	}

	r := api.SetRouter()
	if err := r.Run(":" + viper.GetString("http.port")); err != nil {
		log.Fatalf("Error: Got '%v'", err)
	}
}
