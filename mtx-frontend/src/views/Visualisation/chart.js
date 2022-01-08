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
      isScoring: props.score,
      score_msg: '',
    };
    if (this.state.isScoring) {
      this.setState({ score_msg: 'Score!!!' });
    } else {
      this.setState({ score_msg: 'No Score. Better luck next time' });
    }
  }
  componentDidMount() {
    console.log(this.context);
    const { index, changeIndex } = this.context;
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

    const yAccessor = (d) => d.value;
    const xAccessor = (d) => d['time'];

    let dimensions = {
      width: window.innerWidth * 0.8,
      height: 550,
      margin: {
        top: 115,
        right: 30,
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

    //6. Convert a datapoints into X and Y value

    const lineGenerator = d3
      .line()
      .x((d) => xScale(xAccessor(d)))
      .y((d) => yScale(yAccessor(d)))
      .curve(d3.curveBasis);

    // 7. Convert X and Y into Path
    var div = d3
      .select('body')
      .append('div')
      .attr('class', 'tooltip-donut')
      .style('opacity', 0);

    const line = bounds
      .append('path')
      .attr('d', lineGenerator(dataset))
      .attr('fill', 'none')
      .attr('stroke', 'Red')
      .attr('stroke-width', 2);
    // .on("mouseover", function (d, i) {
    //   d3.select(this).transition().duration("50").attr("opacity", ".2");
    //   //Makes the new div appear on hover:
    //   div.transition().duration(50).style("opacity", 1);
    //   d3.pointer("mouseover", function (d, i, e) {
    //     console.log(d, i, e);
    //   });
    //   let num = Math.round((d.value / d.data.all) * 100).toString() + "%";
    //   div.html(num);
    //   // .style("left", d3.event.pageX + 10 + "px")
    //   // .style("top", d3.event.pageY - 15 + "px");
    // })
    // .on("mouseout", function (d, i) {
    //   d3.select(this).transition().duration("50").attr("opacity", "1");
    //   //Makes the new div disappear:
    //   div.transition().duration("50").style("opacity", 0);
    // });

    //8. Create X axis and Y axis
    // Generate Y Axis

    const yAxisGenerator = d3.axisLeft().scale(yScale);
    const yAxis = bounds.append('g').call(yAxisGenerator);
    wrapper
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0)
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
      .text('Frames');

    wrapper
      .append('g')
      .style('transform', `translate(${50}px,${15}px)`)
      .append('text')
      .attr('class', 'title')
      .attr('x', dimensions.width / 2)
      .attr('y', dimensions.margin.top / 2)
      .attr('text-anchor', 'middle')
      .text('Probability Chart')
      .style('font-size', '24px')
      // .style("text-decoration", "underline")
      .style('color', 'red');

    wrapper
      .append('g')
      .style('transform', `translate(${50}px,${30}px)`)
      .append('text')
      .attr('class', 'score-message')
      .attr('x', dimensions.width / 2)
      .attr('y', dimensions.margin.top / 2)
      .attr('text-anchor', 'middle')
      .text(this.state.score_msg)
      .style('font-size', '24px')
      // .style("text-decoration", "underline")
      .style('color', 'red');
  }

  render() {
    return <div id='wrapper'></div>;
  }
}

ChartControls.contextType = IndexContext;

export default ChartControls;
