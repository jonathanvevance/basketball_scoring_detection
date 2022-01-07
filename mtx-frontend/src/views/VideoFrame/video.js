import React, { Component } from "react";
import ShowVideo from "./videocontrols";
import VideoData from "./videodata";
import VideoContext, { VideoContextProvider } from "./videocontext";
import { Typography, Grid } from "@material-ui/core";
import "bootstrap/dist/css/bootstrap.min.css";
import rim from "../../assets/rim.jpeg";
import gif from "../../assets/stephen-curry-basketball.gif";
class VideoUI extends Component {
  constructor(props) {
    super(props);
    this.state = {
      frame: "0",
      totalTime: "0",
      currentTime: "0",
    };
  }
  render() {
    return (
      <div className="vidAnalyserBody">
        <div className="rimBg">
          <img src={rim} alt="" />
        </div>
        <VideoContextProvider>
          <Grid
            container
            spacing={0}
            style={{
              // padding: "10px",
              // margin: "20px 0",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              height: "100%",
              textAlign: "center",
            }}
          >
            <Grid item xs={6} className="video">
              <ShowVideo />
            </Grid>
            <Grid item xs={6} className="video-data">
              <VideoData />
            </Grid>
          </Grid>
        </VideoContextProvider>
      </div>
    );
  }
}

export default VideoUI;
