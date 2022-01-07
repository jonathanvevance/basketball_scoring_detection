import React, { Component } from 'react';
import ShowVideo from './videocontrols';
import VideoData from './videodata';
import VideoContext, { VideoContextProvider } from './videocontext';
import { Typography, Grid } from '@material-ui/core';

class VideoUI extends Component {
  constructor(props) {
    super(props);
    this.state = {
      frame: '0',
      totalTime: '0',
      currentTime: '0',
    };
  }
  render() {
    return (
      <div>
        <VideoContextProvider>
          <Grid
            container
            spacing={0}
            style={{
              padding: '10px',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              height: '100%',
            }}
          >
            <Grid item xs={6} className='video'>
              <ShowVideo />
            </Grid>
            <Grid item xs={6} className='video-data'>
              <VideoData />
            </Grid>
          </Grid>
        </VideoContextProvider>
      </div>
    );
  }
}

export default VideoUI;
