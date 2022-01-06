import React from "react";
import { Router, Route, Switch } from "react-router-dom";
import { createBrowserHistory } from "history";
import "../App.css";
import NavBar from "../components/NavBar";

import Predict from "../views/Predict/predict";
import Results from "../views/Results/results";
import VisualsUI from "../views/Visualisation/visuals";

var hist = createBrowserHistory();

var routes = [
  { path: "/", component: Predict },
  { path: "/results", component: Results },
  { path: "/charts", component: VisualsUI },
];

export default class App extends React.Component {
  state = {
    loading: true,
  };

  componentDidMount() {
    this.fakeRequest().then(() => {
      const ball = document.querySelector(".loader");
      if (ball) {
        ball.remove(); // removing the spinner element
        this.setState({ loading: false }); // home page displays
      }
    });
  }

  fakeRequest = () => {
    return new Promise((resolve) => setTimeout(() => resolve(), 3000));
  };

  render() {
    if (this.state.loading) {
      return null; //app is not ready (fake request is in process)
    }

    return (
      <div>
        <NavBar />
        <div>
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
