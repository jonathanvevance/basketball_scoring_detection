import React from 'react';
import * as d3 from 'd3';
// import { LineChart } from '@d3/line-chart';

const margin = { top: 40, right: 80, bottom: 60, left: 50 },
  width = 960 - margin.left - margin.right,
  height = 280 - margin.top - margin.bottom,
  color = 'OrangeRed';

// const chart = (data) => {
//   const [activeIndex, setActiveIndex] = React.useState(null),
//     [data, setData] = React.useState([]);

//   const yMinValue = d3.min(data, (d) => d.value),
//     yMaxValue = d3.max(data, (d) => d.value);

//   const getX = d3
//     .scaleTime()
//     .domain(d3.extent(data, (d) => d.time))
//     .range([0, width]);

//   const getY = d3
//     .scaleLinear()
//     .domain([yMinValue - 1, yMaxValue + 2])
//     .range([height, 0]);

//   const getXAxis = (ref) => {
//     const xAxis = d3.axisBottom(getX);
//     d3.select(ref).call(xAxis.tickFormat(d3.timeFormat('%b')));
//   };

//   const getYAxis = (ref) => {
//     const yAxis = d3.axisLeft(getY).tickSize(-width).tickPadding(7);
//     d3.select(ref).call(yAxis);
//   };
// };

class ChartControls extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      data: [],
      options: [],
    };
  }
  componentDidMount() {
    // axios.get('http://localhost:5000/getprobab').then((result) => {
    //   console.log(result.data.data);
    //   this.setState({ data: result.data.data });
    // });
    this.drawLineChart();
  }
  async drawLineChart() {
    const dataset = await d3.json('http://localhost:4000/getvalue');
    // console.log(dataset);
    const yAccessor = (d) => d.value;
    const xAccessor = (d) => d['time'];
    console.log(xAccessor(dataset[2]));

    let dimensions = {
      width: window.innerWidth * 0.6,
      height: 600,
      margin: {
        top: 115,
        right: 20,
        bottom: 40,
        left: 60,
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
    console.log(wrapper);

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

    // Generate X Axis
    const xAxisGenerator = d3.axisBottom().scale(xScale);
    const xAxis = bounds
      .append('g')
      .call(xAxisGenerator.tickFormat(d3.timeFormat('%b')))
      .style('transform', `translateY(${dimensions.boundedHeight}px)`);

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
      .style('text-decoration', 'underline');
  }

  render() {
    return (
      <div id='root'>
        <div id='options'>
          <label>
            <input name='radio' type='radio' />
            2018
          </label>
          <label>
            <input name='radio' type='radio' />
            2019
          </label>
          <label>
            <input name='radio' type='radio' checked />
            2020
          </label>
        </div>
        <div id='wrapper'></div>
      </div>
    );
  }
}

export default ChartControls;
