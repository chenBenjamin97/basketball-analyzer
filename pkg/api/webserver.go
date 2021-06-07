package api

import (
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path"

	"github.com/chenBenjamin97/final-project/pkg/utils"
	"github.com/chenBenjamin97/final-project/pkg/video"
	"github.com/gin-gonic/gin"
	"github.com/spf13/viper"
)

func SetRouter() *gin.Engine {
	r := gin.Default()

	//serve html pages to client
	r.Static("/client", viper.GetString("frontend.static-files-path"))
	r.StaticFile("/", viper.GetString("frontend.static-files-path")+"home_page/dist/index.html")

	apiRoutes := r.Group("/api")

	apiRoutes.GET("/ReadyVideosNames", func(ctx *gin.Context) {
		if names, err := utils.ListDir(viper.GetString("directory.ready")); err != nil {
			ctx.Status(http.StatusInternalServerError)
		} else {
			ctx.JSON(http.StatusOK, names)
		}
	})

	apiRoutes.GET("/UserUploadsVideosNames", func(ctx *gin.Context) {
		if names, err := utils.ListDir(viper.GetString("directory.source")); err != nil {
			ctx.Status(http.StatusInternalServerError)
		} else {
			ctx.JSON(http.StatusOK, names)
		}
	})

	apiRoutes.GET("/Play", func(ctx *gin.Context) {
		videoName := ctx.Request.URL.Query().Get("name")
		if videoName == "" {
			ctx.Status(http.StatusNotAcceptable) //missing url parameter
			return
		}

		analyzed := ctx.Request.URL.Query().Get("analyzed")
		if analyzed != "true" && analyzed != "false" {
			ctx.Status(http.StatusNotAcceptable) //missing url parameter
			return
		}

		var videoPath string
		if analyzed == "true" {
			videoPath = path.Join(viper.GetString("directory.ready"), videoName+"."+viper.GetString("video.prod_format"))
		} else {
			videoPath = path.Join(viper.GetString("directory.source"), videoName+"."+viper.GetString("video.prod_format"))
		}

		if _, err := os.Stat(videoPath); err != nil {
			if os.IsNotExist(err) {
				ctx.Status(http.StatusNotFound)
				return
			} else {
				ctx.Status(http.StatusInternalServerError)
				return
			}
		}

		ctx.Header("Content-Type", "video/mp4")
		http.ServeFile(ctx.Writer, ctx.Request, videoPath)
	})

	apiRoutes.POST("/Upload", func(ctx *gin.Context) {
		// ctx.Request.ParseMultipartForm(15 << 20) //limit file size at body to 15MB
		file, fHeader, err := ctx.Request.FormFile("video")
		if err != nil {
			ctx.Status(http.StatusInternalServerError)
			return
		}

		if existNames, err := utils.ListDir(viper.GetString("directory.source")); err != nil {
			ctx.Status(http.StatusInternalServerError)
			return
		} else {
			if utils.InSlice(fHeader.Filename, existNames) {
				ctx.Status(http.StatusNotAcceptable)
				return
			}
		}

		defer file.Close()
		log.Printf("api/Upload: Recived new file: name - '%s', size - %v Bytes", fHeader.Filename, fHeader.Size)

		fileBytes, err := ioutil.ReadAll(file)
		if err != nil {
			log.Printf("api/Upload: Could not read request's body, got '%v'", err)
			ctx.Status(http.StatusInternalServerError)
			return
		}

		srcFilePath := path.Join(viper.GetString("directory.source"), fHeader.Filename)

		if err = ioutil.WriteFile(srcFilePath, fileBytes, 0444); err != nil {
			log.Printf("api/Upload: Could not write '%s' file, got '%v'", srcFilePath, err)
			ctx.Status(http.StatusInternalServerError)
			return
		}

		go video.Tag(fHeader.Filename)
	})

	return r
}
