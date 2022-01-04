import React, { Component } from 'react';
import Select from 'react-select';
import makeAnimated from 'react-select/animated';
import 'bootstrap/dist/css/bootstrap.min.css';

const animatedComponents = makeAnimated();

const Countries = [
  { label: 'Full', value: 1 },
  { label: 'First 40', value: 2 },
  { label: 'Last 60', value: 3 },
  { label: 'Last 40', value: 4 },
  { label: 'Last 20', value: 5 },
];

class SimpleListMenu extends Component {
  componentDidMount() {
    this.props.parentCallback(1);
  }
  onChangeCallback = (event) => {
    this.props.parentCallback(event.value);
    console.log(event.value);
  };
  render() {
    return (
      <div className='container'>
        <div className='row'>
          <div className='col-md-3'></div>
          <div className='col-md-6'>
            <Select
              options={Countries}
              components={animatedComponents}
              onChange={(event) => this.onChangeCallback(event)}
            />
          </div>
          <div className='col-md-4'></div>
        </div>
      </div>
    );
  }
}

export default SimpleListMenu;
