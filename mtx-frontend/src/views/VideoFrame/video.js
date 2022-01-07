import React, { useState } from 'react';
import VideoPlayer from 'react-video-player-extended';
import './styles.css';

function VideoUI() {
  const [url] = useState('http://localhost:5000/video');
  const [settings, setSettings] = useState(['Title', 'FPS']);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(0.7);
  const [timeStart] = useState(0);
  const [fps] = useState(30);

  const controls = [
    'Play',
    'Time',
    'Progress',
    'Volume',
    'FullScreen',
    'NextFrame',
    'LastFrame',
  ];

  const handlePlay = () => {
    setIsPlaying(true);
  };

  const handlePause = () => {
    setIsPlaying(false);
  };

  const handleVolume = (value) => {
    setVolume(value);
  };

  const handleProgress = (e) => {
    console.log('Current time: ', e.target.currentTime);
  };

  const handleDuration = (duration) => {
    console.log('Duration: ', duration);
  };

  return (
    <div className='container'>
      <header className='main-header'>
        <h1 className='app-name'>Video Analyser</h1>
      </header>
      <VideoPlayer
        url={url}
        controls={controls}
        isPlaying={isPlaying}
        volume={volume}
        loop={true}
        height={'auto'}
        width={'640px'}
        timeStart={timeStart}
        onPlay={handlePlay}
        onPause={handlePause}
        onVolume={handleVolume}
        onProgress={handleProgress}
        onDuration={handleDuration}
        fps={fps}
        viewSettings={settings}
      />
      <div className='controls'>
        {/* <p className='control-list'>
          Controls:
          {controlsList.map((control) => {
            return (
              <label key={control.id.toString()} htmlFor={control.id}>
                <input
                  id={control.id}
                  type='checkbox'
                  checked={controls.includes(control.id)}
                  onChange={handleControlToggle}
                />{' '}
                {control.title}
              </label>
            );
          })}
        </p> */}
        {/* <p className='control-list'>
          State:
          <span style={{ height: 3 }} />
          controls: {controls.join(', ')}
          <span style={{ height: 3 }} />
          isPlaying: {isPlaying.toString()}
          <span style={{ height: 3 }} />
          volume: {volume}
          <span style={{ height: 3 }} />
          timeStart: {timeStart}
          <span style={{ height: 3 }} />
          fps: {fps}
        </p> */}
      </div>
    </div>
  );
}

// export default App

export default VideoUI;
