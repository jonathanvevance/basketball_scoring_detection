import React from 'react';
import * as d3 from 'd3';

export default class BarChart extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      data: props.barData,
      range: '1M',
      width: props.width,
    };
  }
  componentDidMount() {
    const datas = this.state.data;
    const title = datas.title;
    console.log(datas.length);
    this.drawBarChart(datas, title);
  }

  getTitle = (d) =>
    [
      `${d3.timeFormat('%m/%d/%Y')(d.Date)}`,
      `Open: ${d.Open}`,
      `Close: ${d.Close}`,
      `High: ${d.High}`,
      `Low: ${d.Low}`,
      `Volume: ${d3.format(',')(d.Volume)}`,
    ].join(' ');

  getColor = (d) => (d.Close >= d.Open ? '#649334' : '#cc392b');
  firstDays = () => {
    const dates = [];
    let p = -1;
    this.datas.reverse().forEach((d) => {
      if (d.Date.getMonth() !== p) dates.push(d.Date);
      p = d.Date.getMonth();
    });
    return dates;
  };
  ticks = (d) => {
    if (this.state.range === '5D' || this.state.range === '1M') {
      return d3.timeFormat('%m-%d')(d);
    } else if (this.firstDays.indexOf(d) >= 0) {
      if (d.getMonth() === 0 || d === this.state.data[0].Date) {
        return d.getFullYear();
      } else if (this.state.range === '5Y' || this.state.range === 'Max') {
        return d.getMonth() === 0 || d === this.state.data[0].Date
          ? d.getFullYear()
          : '';
      } else {
        return d3.timeFormat('%B')(d).substring(0, 3);
      }
    }
    return '';
  };

  enableOverlay = (overlay) => {
    [...overlay.querySelectorAll('g[text-anchor]')].forEach((g) => g.remove());
    const defaultTitle = 'TSLA Historical Data';
    const text = d3
      .select(overlay)
      .append('text')
      .attr('dy', '1em')
      .attr('text-anchor', 'start')
      .attr('font-weight', 'bold')
      .text(defaultTitle);

    const rect = d3
      .select(overlay)
      .style('position', 'absolute')
      .style('top', 0)
      .style('background-color', 'transparent')
      .selectAll('rect');

    rect
      .attr('fill', '#aaa')
      .attr('fill-opacity', 0)
      .on('pointerover', (e) => {
        const i = rect.nodes().indexOf(e.currentTarget);
        const d = this.state.data[i];
        e.currentTarget.setAttribute('fill-opacity', 0.5);
        text.attr('fill', (_) => this.getColor(d)).text(this.getTitle(d));
      })
      .on('pointerout', (e) => {
        e.currentTarget.setAttribute('fill-opacity', 0);
        text.attr('fill', 'currentColor').text(defaultTitle);
      });
  };
  drawBarChart(data, title) {
    const canvasHeight = 400;
    const canvasWidth = 600;
    const scale = 1;
    const margin = { top: 30, right: 0, bottom: 0, left: 40 };

    const x = d3
      .scaleBand()
      .domain(d3.range(data.length))
      .range([margin.left, canvasWidth - margin.right])
      .padding(0.1);
    const y = d3
      .scaleLinear()
      .domain([0, d3.max(data, (d) => d.value)])
      .nice()
      .range([canvasHeight - margin.bottom, margin.top]);
    const yAxis = (g) =>
      g
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(y).ticks(null, data.format))
        .call((g) => g.select('.domain').remove())
        .call((g) =>
          g
            .append('text')
            .attr('x', -margin.left)
            .attr('y', 10)
            .attr('fill', 'currentColor')
            .attr('text-anchor', 'start')
            .text(data.y)
        );
    const xAxis = (g) =>
      g.attr('transform', `translate(0,${canvasHeight - margin.bottom})`).call(
        d3
          .axisBottom(x)
          .tickFormat((i) => data[i].name)
          .tickSizeOuter(0)
      );

    const svgCanvas = d3
      .select(this.refs.canvas)
      .append('svg')
      .attr('width', canvasWidth)
      .attr('height', canvasHeight)
      .attr('class', 'tile')
      .attr('viewBox', [0, 0, canvasWidth * scale, canvasHeight * scale]);
    // .style('border', '1px solid black');
    svgCanvas
      .append('g')
      .attr('fill', 'red')
      .selectAll('rect')
      .data(data)
      .join('rect')
      .attr('x', (d, i) => x(i))
      .attr('y', (d) => y(d.value))
      .attr('height', (d) => y(0) - y(d.value))
      .attr('width', x.bandwidth())
      .append('title')
      .text((d) => `${d.value} `);
    svgCanvas
      .append('g')
      .selectAll('text')
      .data(data)
      .join('text')
      .attr('x', (d, i) => x(i) + x.bandwidth() / 2)
      .attr('y', (d) => y(d.value))
      .attr('text-anchor', 'middle')
      .style('font-size', '13px')
      .style('font-family', 'Proxima Reg')
      .text((d) => `${d.value} `);
    svgCanvas
      .append('text')
      .attr('x', canvasWidth / 2)
      .attr('y', 0 - margin.top / 2)
      .attr('text-anchor', 'middle')
      .style('font-size', '16px')
      .style('font-family', 'Proxima Reg')
      .style('text-decoration', 'underline')
      .text(data[0].title);

    svgCanvas.append('g').call(xAxis);

    svgCanvas.append('g').call(yAxis);
  }

  render() {
    return (
      // <div class='tile'>
      <div ref='canvas'></div>
      // </div>
    );
  }
}
