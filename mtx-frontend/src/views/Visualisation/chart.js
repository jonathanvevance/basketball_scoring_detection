import React from 'react';
import * as d3 from 'd3';
import './chart.css';
import IndexContext from './indexcontext';

class ChartControls extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      data: props.data,
      options: [],
      newData: null,
    };
  }
  componentDidMount() {
    console.log(this.context);
    const { index, changeIndex } = this.context;
    // this.setState({ newData: this.changeData(this.context.index) });
    // console.log(this.changeData(index));
    // console.log(this.state.newData);
    this.drawLineChart(this.changeData(index));
  }
  changeData = (index) => {
    const oldData = this.state.data;
    if (index === '1') return this.state.data;
    else if (index === '2') {
      let data = [];
      for (let i = 0; i < (this.state.data.length * 2) / 5; i++) {
        data.push(this.state.data[i]);
      }
      return data;
    } else if (index === '3') {
      let data = [];
      for (
        let i = (this.state.data.length * 2) / 5;
        i < this.state.data.length;
        i++
      ) {
        data.push(this.state.data[i]);
      }
      return data;
    } else if (index === '4') {
      let data = [];
      for (
        let i = (this.state.data.length * 3) / 5;
        i < this.state.data.length;
        i++
      ) {
        data.push(this.state.data[i]);
      }
      return data;
    } else if (index === '5') {
      let data = [];
      for (
        let i = (this.state.data.length * 4) / 5;
        i < this.state.data.length;
        i++
      ) {
        data.push(this.state.data[i]);
      }
      return data;
    }
  };
  async drawLineChart(dataset) {
    let activeIndex = null;

    console.log(dataset);
    const yAccessor = (d) => d.value;
    const xAccessor = (d) => d['time'];
    // console.log(xAccessor(dataset[2]));

    let dimensions = {
      width: window.innerWidth * 0.8,
      height: 600,
      margin: {
        top: 115,
        right: 20,
        bottom: 40,
        left: 100,
      },
    };
    dimensions.boundedWidth =
      dimensions.width - dimensions.margin.left - dimensions.margin.right;
    dimensions.boundedHeight =
      dimensions.height - dimensions.margin.top - dimensions.margin.bottom;

    const wrapper = d3
      .select('#wrapper')
      .append('svg')
      .attr('width', dimensions.width)
      .attr('height', dimensions.height);

    //Log our new Wrapper Variable to the console to see what it looks like
    // console.log(wrapper);

    // 4. Create a Bounding Box

    const bounds = wrapper
      .append('g')
      .style(
        'transform',
        `translate(${dimensions.margin.left}px,${dimensions.margin.top}px)`
      );

    // 5. Define Domain and Range for Scales

    const xScale = d3
      .scaleTime()
      .domain(d3.extent(dataset, xAccessor))
      .range([0, dimensions.boundedWidth]);

    const yScale = d3
      .scaleLinear()
      .domain(d3.extent(dataset, yAccessor))
      .range([dimensions.boundedHeight, 0]);

    const referenceBandPlacement = yScale(0.2);
    const referenceBand = bounds
      .append('rect')
      .attr('x', 0)
      .attr('width', dimensions.boundedWidth)
      .attr('y', referenceBandPlacement)
      .attr('height', dimensions.boundedHeight - referenceBandPlacement)
      .attr('fill', '#ffece6');

    //6. Convert a datapoints into X and Y value

    const lineGenerator = d3
      .line()
      .x((d) => xScale(xAccessor(d)))
      .y((d) => yScale(yAccessor(d)))
      .curve(d3.curveBasis);

    // 7. Convert X and Y into Path

    const line = bounds
      .append('path')
      .attr('d', lineGenerator(dataset))
      .attr('fill', 'none')
      .attr('stroke', 'Red')
      .attr('stroke-width', 2);

    //8. Create X axis and Y axis
    // Generate Y Axis

    const yAxisGenerator = d3.axisLeft().scale(yScale);
    const yAxis = bounds.append('g').call(yAxisGenerator);
    wrapper
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - dimensions.margin.left)
      .attr('x', 0 - dimensions.height / 2)
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .text('Probability');

    // Generate X Axis
    const xAxisGenerator = d3.axisBottom().scale(xScale);
    const xAxis = bounds
      .append('g')
      .call(xAxisGenerator.tickFormat(d3.timeFormat('%s')))
      .style('transform', `translateY(${dimensions.boundedHeight}px)`);
    wrapper
      .append('text') // text label for the x axis
      .attr(
        'transform',
        'translate(' +
          dimensions.width / 2 +
          ' ,' +
          (dimensions.height + dimensions.margin.bottom) +
          ')'
      )
      .style('text-anchor', 'middle')
      .text('Time');

    wrapper
      .append('g')
      .style('transform', `translate(${50}px,${15}px)`)
      .append('text')
      .attr('class', 'title')
      .attr('x', dimensions.width / 2)
      .attr('y', dimensions.margin.top / 2)
      .attr('text-anchor', 'middle')
      .text('Probability Chart')
      .style('font-size', '36px')
      .style('text-decoration', 'underline')
      .style('color', 'red');
  }

  render() {
    return <div id='wrapper'></div>;
  }
}

ChartControls.contextType = IndexContext;

export default ChartControls;
