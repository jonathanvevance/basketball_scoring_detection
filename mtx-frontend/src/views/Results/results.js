import React, { Component } from 'react';
import { Button } from '@material-ui/core';
import BarChart from './barChart';
import './results.css';

const barData = [
  { name: 'E', value: 0.12702, title: 'Graph 1' },
  { name: 'T', value: 0.09056 },
  { name: 'A', value: 0.08167 },
  { name: 'O', value: 0.07507 },
  { name: 'I', value: 0.06966 },
  { name: 'N', value: 0.06749 },
  { name: 'S', value: 0.06327 },
  { name: 'H', value: 0.06094 },
  { name: 'R', value: 0.05987 },
  { name: 'D', value: 0.04253 },
  { name: 'L', value: 0.04025 },
  { name: 'C', value: 0.02782 },
  { name: 'U', value: 0.02758 },
  { name: 'M', value: 0.02406 },
  { name: 'W', value: 0.0236 },
  { name: 'F', value: 0.02288 },
  { name: 'G', value: 0.02015 },
  { name: 'Y', value: 0.01974 },
  { name: 'P', value: 0.01929 },
  { name: 'B', value: 0.01492 },
];

class Results extends Component {
  constructor(props) {
    super(props);
    // this.selectFile = this.selectFile.bind(this);
    // this.upload = this.upload.bind(this);

    this.state = {
      barData: barData,
      pieData: [],
      graphOn: false,
      currentFile: undefined,
      previewImage: undefined,
      progress: 0,

      message: '',
      isError: false,
      loader: true,
    };
  }
  render() {
    return (
      <div className='overall'>
        <div className='top grid-layout'>
          <div className='tile-left'></div>
          <div className='tile-middle'>
            <h1>Results</h1>
          </div>
          <div className='tile-right'></div>
        </div>
        <div className='bottom'>
          <div className='centered'>
            <div className='grid-layout'>
              <BarChart barData={barData} />
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default Results;
