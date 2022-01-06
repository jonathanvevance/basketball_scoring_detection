import React, { Component } from "react";
import SimpleListMenu from "./dropmenu";
import * as d3 from "d3";
import ChartControls from "./chart";
import "bootstrap/dist/css/bootstrap.min.css";
import IndexContext, { IndexProvider } from "./indexcontext";
import loader from "../../assets/loader.png";
class VisualsUI extends Component {
  constructor(props) {
    super(props);
    this.state = {
      lineData: [],
      index: 0,
      dropIndex: 1,
      isLoading: true,
    };
  }

  callbackFunction = (childData) => {
    this.setState({ dropIndex: childData });
  };

  async getChartData() {
    const dataset = await d3
      .json("http://localhost:4000/getvalue")
      .then((d) => {
        const parseDate = d3.timeParse("%s");
        d.forEach((i) => {
          i.time = parseDate(i.time);
          i.value = Number(i.value);
        });
        return d;
      });
    this.setState({ lineData: dataset, isLoading: false });
  }

  componentDidMount() {
    this.getChartData();
  }

  render() {
    return (
      <div>
        <IndexProvider>
          <div className="container">
            <div className="row">
              <div
                style={{
                  marginTop: "30px",
                }}
              >
                <SimpleListMenu parentCallback={this.callbackFunction} />
              </div>
              <div>
                {this.state.isLoading ? (
                  // <div class="loader">
                  //   <img class="ball" src={loader} alt="" />
                  // </div>
                  <div class="chartLoading">
                    <div class="bar1"></div>
                    <div class="bar2"></div>
                    <div class="bar3"></div>
                    <div class="bar4"></div>
                  </div>
                ) : (
                  this.state.lineData.length > 0 && (
                    <ChartControls
                      key={this.state.dropIndex}
                      data={this.state.lineData}
                    />
                  )
                )}
              </div>
            </div>
          </div>
        </IndexProvider>
      </div>
    );
  }
}

export default VisualsUI;
