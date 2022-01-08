import React, { Component } from 'react';
import ShowVideo from './videocontrols';
import VideoData from './videodata';
import VideoContext, { VideoContextProvider } from './videocontext';
import { Typography, Grid } from '@material-ui/core';
import { json, timeParse } from 'd3';
import 'bootstrap/dist/css/bootstrap.min.css';
import rim from '../../assets/rim.jpeg';
import gif from '../../assets/stephen-curry-basketball.gif';
class VideoUI extends Component {
  constructor(props) {
    super(props);
    this.state = {
      frame: '0',
      totalTime: '0',
      currentTime: '0',
      callbackTime: '0',
      lineData: [],
      fps: 30,
      is_scoring: false,
      scoreMessage: 'Scroll video to find status',
    };
  }
  componentDidMount() {
    this.getChartData();
  }
  async getChartData() {
    const dataset = await json('http://localhost:4000/getvideodata').then(
      (d) => {
        const parseDate = timeParse('%s');
        d.forEach((i) => {
          i.time = Number(i.time);
          i.value = Number(i.value);
          i.fps = Number(i.fps);
        });
        return d;
      }
    );
    console.log(dataset);
    console.log('fps: ', dataset[0]['fps']);
    this.setState({ lineData: dataset });
    this.setState({ fps: dataset[0]['fps'] });
    this.setState({ is_scoring: dataset[0]['is_scoring'] });
    console.log('is_scoring: ', this.state.is_scoring);
    if (this.state.is_scoring === 'True') {
      this.setState({ scoreMessage: 'Score!!!' });
    } else {
      this.setState({ scoreMessage: 'Not Scoring :(' });
    }
  }

  progressCallback = (childData) => {
    this.setState({ callbackTime: childData });
  };

  render() {
    return (
      <div className='vidAnalyserBody'>
        <div className='rimBg'>
          <img src={rim} alt='' />
        </div>
        <VideoContextProvider>
          <Grid
            container
            spacing={0}
            style={{
              // padding: "10px",
              // margin: "20px 0",
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              height: '100%',
              textAlign: 'center',
            }}
          >
            <Grid item xs={6} className='video'>
              {this.state.lineData.length > 0 && (
                <ShowVideo
                  parentCallback={this.progressCallback}
                  fps={this.state.fps}
                />
              )}
            </Grid>
            <Grid item xs={6} className='video-data'>
              {this.state.lineData.length > 0 && (
                <VideoData
                  key={this.state.callbackTime}
                  data={this.state.lineData}
                  fps={this.state.fps}
                  scoreMessage={this.state.scoreMessage}
                />
              )}
            </Grid>
          </Grid>
        </VideoContextProvider>
      </div>
    );
  }
}

export default VideoUI;
