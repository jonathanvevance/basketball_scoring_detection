import React, { Component } from 'react';
import VideoContext from './videocontext';
import { json, timeParse } from 'd3';
import { Typography } from '@material-ui/core';

class VideoData extends Component {
  constructor(props) {
    super(props);
    this.state = {
      dataset: [],
      frameProbab: 0,
      frame: 0,
    };
  }
  componentDidMount() {
    console.log(this.context);
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
    this.setState({ lineData: dataset });
  }
  render() {
    return (
      <div>
        <Typography variant='h4'>Video Data</Typography>
        <Typography variant='h6'>Frame: {this.state.frame}</Typography>
        <Typography variant='h6'>
          Frame Probability: {this.state.frameProbab}
        </Typography>
      </div>
    );
  }
}
VideoData.contextType = VideoContext;

export default VideoData;
