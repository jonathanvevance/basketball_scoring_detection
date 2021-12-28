import React from 'react';
import { Chart } from 'react-google-charts';

export const data = [
  ['Date', 'Value'],
  [new Date(1996, 1, 1), 2000 * Math.random()],
  [new Date(1997, 1, 1), 2000 * Math.random()],
  [new Date(1998, 1, 1), 2000 * Math.random()],
  [new Date(1999, 1, 1), 2000 * Math.random()],
  [new Date(2000, 1, 1), 2000 * Math.random()],
  [new Date(2001, 1, 1), 2000 * Math.random()],
  [new Date(2002, 1, 1), 2000 * Math.random()],
  [new Date(2003, 1, 1), 2000 * Math.random()],
  [new Date(2004, 1, 1), 2000 * Math.random()],
  [new Date(2005, 1, 1), 2000 * Math.random()],
  [new Date(2006, 1, 1), 2000 * Math.random()],
  [new Date(2007, 1, 1), 2000 * Math.random()],
  [new Date(2008, 1, 1), 2000 * Math.random()],
  [new Date(2009, 1, 1), 2000 * Math.random()],
];

export const options = {
  chartArea: { height: '80%', width: '90%' },
  hAxis: { slantedText: false },
  vAxis: { viewWindow: { min: 0, max: 2000 } },
  legend: { position: 'none' },
};

export function ChartControls() {
  return (
    <Chart
      chartType='LineChart'
      width='80%'
      height='400px'
      data={data}
      options={options}
      chartPackages={['corechart', 'controls']}
      controls={[
        {
          controlType: 'DateRangeFilter',
          options: {
            filterColumnLabel: 'Date',
            ui: { format: { pattern: 'yyyy' } },
          },
        },
      ]}
    />
  );
}
