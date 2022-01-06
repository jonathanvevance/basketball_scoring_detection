import React, { Component } from 'react';
import SimpleListMenu from './dropmenu';
import * as d3 from 'd3';
import ChartControls from './chart';
import IndexContext, { IndexProvider } from './indexcontext';
class VisualsUI extends Component {
  constructor(props) {
    super(props);
    this.state = {
      lineData: [],
      index: 0,
      dropIndex: 1,
    };
  }

  callbackFunction = (childData) => {
    this.setState({ dropIndex: childData });
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
  }

  componentDidMount() {
    this.getChartData();
  }

  render() {
    return (
      <div>
        <IndexProvider>
          <SimpleListMenu parentCallback={this.callbackFunction} />
          {this.state.lineData.length > 0 && (
            <ChartControls
              key={this.state.dropIndex}
              data={this.state.lineData}
            />
          )}
        </IndexProvider>
      </div>
    );
  }
}

export default VisualsUI;
