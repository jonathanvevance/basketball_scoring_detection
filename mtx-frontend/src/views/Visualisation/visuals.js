import React, { Component, useCallback } from 'react';
import SimpleListMenu from './dropmenu';
import * as d3 from 'd3';
import { index } from 'd3';
class VisualsUI extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      lineData: [],
      index: 0,
    };
  }
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
  }

  componentDidMount() {
    this.getChartData();
  }

  render() {
    return (
      <div>
        <SimpleListMenu lineData={this.state.lineData} />
      </div>
    );
  }
}

export default VisualsUI;
