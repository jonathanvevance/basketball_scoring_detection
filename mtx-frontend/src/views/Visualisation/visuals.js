import React, { Component } from 'react';
import SimpleListMenu from './dropmenu';
import * as d3 from 'd3';
import ChartControls from './chart';
class VisualsUI extends Component {
  constructor(props) {
    super(props);
    this.state = {
      lineData: [],
      index: 0,
      dropIndex: 1,
      newData: [],
    };
  }
  callbackFunction = (childData) => {
    this.setState({ dropIndex: childData });
    console.log(this.state.dropIndex);
    this.setState({ newData: this.changeData(this.state.dropIndex) });
  };
  async getChartData() {
    const dataset = await d3
      .json('http://localhost:4000/getvalue')
      .then((d) => {
        const parseDate = d3.timeParse('%s');
        d.forEach((i) => {
          i.time = parseDate(i.time);
          i.value = Number(i.value);
        });
        return d;
      });
    this.setState({ lineData: dataset });
    this.setState({ newData: dataset });
  }

  componentDidMount() {
    this.getChartData();
  }

  changeData = (index) => {
    if (index === 1) return this.state.lineData;
    else if (index === 2) {
      let data = [];
      for (let i = 0; i < (this.state.lineData.length * 2) / 5; i++) {
        data.push(this.state.lineData[i]);
      }
      return data;
    } else if (index === 3) {
      let data = [];
      for (
        let i = (this.state.lineData.length * 2) / 5;
        i < this.state.lineData.length;
        i++
      ) {
        data.push(this.state.lineData[i]);
      }
      return data;
    } else if (index === 4) {
      let data = [];
      for (
        let i = (this.state.lineData.length * 3) / 5;
        i < this.state.lineData.length;
        i++
      ) {
        data.push(this.state.lineData[i]);
      }
      return data;
    } else if (index === 5) {
      let data = [];
      for (
        let i = (this.state.lineData.length * 4) / 5;
        i < this.state.lineData.length;
        i++
      ) {
        data.push(this.state.lineData[i]);
      }
      return data;
    }
  };

  render() {
    return (
      <div>
        <SimpleListMenu parentCallback={this.callbackFunction} />
        {this.state.newData.length > 0 && (
          <ChartControls key={this.state.dropIndex} data={this.state.newData} />
        )}
      </div>
    );
  }
}

export default VisualsUI;
