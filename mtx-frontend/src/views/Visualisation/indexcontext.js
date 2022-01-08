/**
 * @file React Context for Dropdown and chart
 * @author Adil Mohammed
 * @copyright Adil Mohammed 2022
 * @license MIT
 */

import React, { Component, useContext } from 'react';

const IndexContext = React.createContext();
/**
 * Provides the COntext for dropdown and chart
 * @extends ParentClassNameHereIfAny
 */

export class IndexProvider extends Component {
  state = {
    index: '1',
  };

  changeIndex = (index) => {
    this.setState({ index: index });
  };

  render() {
    const { index } = this.state;
    const { changeIndex } = this;
    return (
      <IndexContext.Provider value={{ index, changeIndex }}>
        {this.props.children}
      </IndexContext.Provider>
    );
  }
}

export default IndexContext;
