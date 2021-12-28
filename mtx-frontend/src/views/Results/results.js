import React, { Component } from 'react';
import { Button } from '@material-ui/core';
import BarChart from './barChart';
import './results.css';
import axios from 'axios';

class Results extends Component {
  constructor(props) {
    super(props);
    // this.selectFile = this.selectFile.bind(this);
    // this.upload = this.upload.bind(this);

    this.state = {
      barData: [],
      graphOn: false,

      message: '',
      isError: false,
      loader: true,
    };
    const barData = [];
  }
  componentDidMount() {
    axios.get('http://localhost:5000/getprobab').then((result) => {
      console.log(result.data);
      // this.setState({ barData: result.data });
      this.barData = result.data.data;
    });
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
              {/* <BarChart barData={this.barData} /> */}
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default Results;
