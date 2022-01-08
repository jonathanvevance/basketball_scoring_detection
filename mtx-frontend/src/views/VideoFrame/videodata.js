import React, { Component } from 'react';
import VideoContext from './videocontext';
import { Typography } from '@material-ui/core';

class VideoData extends Component {
  constructor(props) {
    super(props);
    this.state = {
      dataset: this.props.data,
      frameProbab: 0,
      frame: 0,
      currentTime: 0,
      totalTime: 0,
      fps: this.props.fps,
      scoreMessage: this.props.scoreMessage,
    };
  }
  componentDidMount() {
    console.log(this.context);
    this.setState({ currentTime: this.context.videoTime });
    this.setState({ totalTime: this.context.totalTime });
    this.getFrameData();
  }
  getFrameData = () => {
    console.log(this.context.videoTime, this.state.fps);
    console.log(
      'current frame without ceil(): ',
      this.state.fps * this.context.videoTime
    );
    const currentFrame = Math.ceil(this.state.fps * this.context.videoTime);
    console.log('Current frame: ', currentFrame);
    if (currentFrame < this.state.dataset.length) {
      this.setState({ frame: currentFrame });
      this.setState({ frameProbab: this.state.dataset[currentFrame]['value'] });
    }
  };

  render() {
    return (
      <div className='vidData'>
        {this.state.dataset.length > 0 && (
          <div>
            <Typography
              variant='h4'
              style={{ marginBottom: '10px', fontWeight: '500' }}
            >
              Video Data
            </Typography>
            <Typography variant='h6' style={{ fontWeight: '600' }}>
              Frame: {this.state.frame}
            </Typography>
            <Typography variant='h6' style={{ fontWeight: '600' }}>
              Frame Probability: {this.state.frameProbab}
            </Typography>
            <Typography variant='h7' style={{ fontWeight: '600' }}>
              Status: {this.state.scoreMessage}
            </Typography>
          </div>
        )}
      </div>
    );
  }
}
VideoData.contextType = VideoContext;

export default VideoData;
