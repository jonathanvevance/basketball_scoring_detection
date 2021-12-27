import React from 'react';
import { Router, Route, Switch } from 'react-router-dom';
import { createBrowserHistory } from 'history';

import NavBar from '../components/NavBar';

import Predict from '../views/Predict/predict';
import Results from '../views/Results/results';

var hist = createBrowserHistory();

var routes = [
  { path: '/', component: Predict },
  { path: '/results', component: Results },
];

export default class App extends React.Component {
  render() {
    return (
      <div>
        <NavBar />
        <div style={{ marginTop: '10px' }}>
          <Router history={hist}>
            <Switch>
              {routes.map((prop, key) => {
                return (
                  <Route
                    exact
                    path={prop.path}
                    key={key}
                    component={prop.component}
                  />
                );
              })}
            </Switch>
          </Router>
        </div>
      </div>
    );
  }
}
