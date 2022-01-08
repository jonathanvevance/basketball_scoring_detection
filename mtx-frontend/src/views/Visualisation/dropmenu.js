import React, { Component } from 'react';
import Select from 'react-select';
import makeAnimated from 'react-select/animated';
import 'bootstrap/dist/css/bootstrap.min.css';
import IndexContext from './indexcontext';

const animatedComponents = makeAnimated();

const options = [
  { label: 'Full', value: '1' },
  { label: 'First 40 %', value: '2' },
  { label: 'Last 60 %', value: '3' },
  { label: 'Last 40 %', value: '4' },
  { label: 'Last 20 %', value: '5' },
];

class SimpleListMenu extends Component {
  componentDidMount() {}
  onChangeCallback = (event) => {
    this.props.parentCallback(event.value);
    const { index, changeIndex } = this.context;
    changeIndex(event.value);
  };
  render() {
    return (
      <div className='container'>
        <div className='row'>
          <div className='col-md-3'></div>
          <div className='col-md-6'>
            <Select
              options={options}
              components={animatedComponents}
              onChange={(event) => this.onChangeCallback(event)}
              defaultValue={options[0]}
            />
          </div>
          <div className='col-md-3'></div>
        </div>
      </div>
    );
  }
}
SimpleListMenu.contextType = IndexContext;

export default SimpleListMenu;
