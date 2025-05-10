import React, { Component } from "react";
import * as tf from "@tensorflow/tfjs";
import { loadLayersModel } from "@tensorflow/tfjs-layers";
import gaussian from "gaussian";

import ImageCanvas from "../components/ImageCanvas";
import XYPlot from "../components/XYPlot";
import Explanation from "../components/Explanation";
import { rounder } from "../utils";

import "./App.css";

import encodedData from "../encoded.json";

const MODEL_PATH = process.env.PUBLIC_URL + "/models/generatorjs/model.json";

class App extends Component {
  constructor(props) {
    super(props);
    this.getImage = this.getImage.bind(this);

    this.norm = gaussian(0, 1);

    this.state = {
      model: null,
      digitImg: tf.zeros([28, 28]),
      mu: 0,
      sigma: 0,
      error: null
    };
  }

  componentDidMount() {
    loadLayersModel(MODEL_PATH)
      .then(model => {
        console.log("Model loaded successfully");
        this.setState({ model });
        return this.getImage();
      })
      .then(digitImg => {
        console.log("Image generated successfully");
        this.setState({ digitImg });
      })
      .catch(error => {
        console.error("Error loading model:", error);
        this.setState({ error: error.message });
      });
  }

  async getImage() {
    const { model, mu, sigma } = this.state;
    const zSample = tf.tensor([[mu, sigma]]);
    const output = model.predict(zSample);
    const img = output.mul(tf.scalar(255.0)).reshape([28, 28]);
    // Debug: print min/max values
    output.data().then(data => {
      console.log('Model output range:', Math.min(...data), Math.max(...data));
    });
    return img;
  }

  render() {
    if (this.state.error) {
      return (
        <div className="App">
          <h1>Error Loading Model</h1>
          <p>{this.state.error}</p>
          <p>Please make sure the model files are in the correct location: {MODEL_PATH}</p>
        </div>
      );
    }

    return this.state.model === null ? (
      <div>Loading, please wait</div>
    ) : (
      <div className="App">
        <h1>VAE Latent Space Explorer</h1>
        <div className="ImageDisplay">
          <ImageCanvas
            width={500}
            height={500}
            imageData={this.state.digitImg}
          />
        </div>

        <div className="ChartDisplay">
          <XYPlot
            data={encodedData}
            width={500 - 10 - 10}
            height={500 - 20 - 10}
            xAccessor={d => d[0]}
            yAccessor={d => d[1]}
            colorAccessor={d => d[2]}
            margin={{ top: 20, bottom: 10, left: 10, right: 10 }}
            onHover={({ x, y }) => {
              this.setState({ sigma: y, mu: x });
              this.getImage().then(digitImg => this.setState({ digitImg }));
            }}
          />
        </div>
        <p>Mu: {rounder(this.norm.cdf(this.state.mu), 3)}</p>
        <p>Sigma: {rounder(this.norm.cdf(this.state.sigma), 3)}</p>
        <div className="Explanation">
          <Explanation />
        </div>

        <h5>Created by Taylor Denouden (April 2018)</h5>
      </div>
    );
  }
}

export default App;
