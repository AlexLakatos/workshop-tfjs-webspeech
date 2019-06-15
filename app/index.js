/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as useLoader from '@tensorflow-models/universal-sentence-encoder';
import * as tf from '@tensorflow/tfjs';

import * as d3 from 'd3';

tf.ENV.set('WEBGL_PACK', false);

const EMBEDDING_DIM = 512;

const DENSE_MODEL_URL = './models/intent/model.json';
const METADATA_URL = './models/intent/intent_metadata.json';

const modelUrls = {
  'bidirectional-lstm': './models/bidirectional-tagger/model.json'
};

let use;
async function loadUSE() {
  if (use == null) {
    use = await useLoader.load();
  }
  return use;
}

let intent;
async function loadIntentClassifer(url) {
  if (intent == null) {
    intent = await tf.loadLayersModel(url);
  }
  return intent;
}

let intentMetadata;
async function loadIntentMetadata() {
  if (intentMetadata == null) {
    const resp = await fetch(METADATA_URL);
    intentMetadata = resp.json();
  }
  return intentMetadata;
}

async function classify(sentences) {
  const [use, intent, metadata] = await Promise.all(
    [loadUSE(), loadIntentClassifer(DENSE_MODEL_URL), loadIntentMetadata()]);

  const {
    labels
  } = metadata;
  const activations = await use.embed(sentences);

  const prediction = intent.predict(activations);

  const predsArr = await prediction.array();
  const preview = [predsArr[0].slice()];
  preview.unshift(labels);
  console.table(preview);

  tf.dispose([activations, prediction]);

  return predsArr[0];
}

const THRESHOLD = 0.90;
async function getClassificationMessage(softmaxArr, inputText) {
  const {
    labels
  } = await loadIntentMetadata();
  const max = Math.max(...softmaxArr);
  const maxIndex = softmaxArr.indexOf(max);
  const intentLabel = labels[maxIndex];

  if (max < THRESHOLD) {
    return '¯\\_(ツ)_/¯';
  } else {
    let response;
    switch (intentLabel) {
      case 'GetWeather':
        const model = "bidirectional-lstm";
        var location = await tagMessage(inputText, model);
        if (location.trim() != "") {
          const weatherMessage = await getWeather(location);;
          response = '⛅ ' + weatherMessage;
          speak(weatherMessage);
        } else {
          response = '⛅';
        }
        break;

      case 'PlayMusic':
        response = '🎵🎺🎵';
        break;

      default:
        response = '?';
        break;
    }
    return response;
  }
}

const taggers = {};
/**
 * Load a custom trained token tagger model.
 * @param {string} name Type of model to load. Should be a key in modelUrls
 */
async function loadTagger(name) {
  if (taggers[name] == null) {
    const url = modelUrls[name];
    try {
      taggers[name] = await tf.loadLayersModel(url);
    } catch (e) {
      // Could not load that model. This is not necessarily an error
      // as the user may not have trained all the available model types
      console.log(`Could not load "${name}" model`);
    }
  }
  return taggers[name];
}

/**
 * Load metadata for a model.
 * @param {string} name Name of model. Should be a key in modelUrls
 */
async function loadMetadata(name) {
  const metadataUrl =
    modelUrls[name].replace('model.json', 'tagger_metadata.json');
  try {
    const resp = await fetch(metadataUrl);
    return resp.json();
  } catch (e) {
    // Could not load that model. This is not necessarily an error
    // as the user may not have trained all the available model types
    console.log(`Could not load "${name}" metadata`);
  }
}

/**
 * Load a number of models to allow the browser to cache them.
 */
async function loadTaggerModel() {
  const modelLoadPromises = Object.keys(modelUrls).map(loadTagger);
  return await Promise.all([loadUSE(), ...modelLoadPromises]);
}

/**
 * Split an input string into tokens, we use the same tokenization function
 * as we did during training.
 * @param {string} input
 *
 * @return {string[]}
 */
function tokenizeSentence(input) {
  return input.split(/\b/).map(t => t.trim()).filter(t => t.length !== 0);
}

/**
 * Tokenize a sentence and tag the tokens.
 *
 * @param {string} sentence sentence to tag
 * @param {string} model name of model to use
 *
 * @return {Object} dictionary of tokens, model outputs and embeddings
 */
async function tagTokens(sentence, model = 'bidirectional-lstm') {
  const [use, tagger, metadata] =
  await Promise.all([loadUSE(), loadTagger(model), loadMetadata(model)]);
  const {
    labels,
    sequenceLength
  } = metadata;


  let tokenized = tokenizeSentence(sentence);
  if (tokenized.length > sequenceLength) {
    console.warn(
      `Input sentence has more tokens than max allowed tokens ` +
      `(${sequenceLength}). Extra tokens will be dropped.`);
  }
  tokenized = tokenized.slice(0, sequenceLength);
  const activations = await use.embed(tokenized);

  // get prediction
  const prediction = tf.tidy(() => {
    // Make an input tensor of [1, sequence_len, embedding_size];
    const toPad = sequenceLength - tokenized.length;

    const padTensors = tf.ones([toPad, EMBEDDING_DIM]);
    const padded = activations.concat(padTensors);

    const batched = padded.expandDims();
    return tagger.predict(batched);
  });


  // Prediction data
  let predsArr = (await prediction.array())[0];

  // Add padding 'tokens' to the end of the values that will be displayed
  // in the UI. These are there for illustration.
  if (tokenized.length < sequenceLength) {
    tokenized.push(labels[2]);
    predsArr = predsArr.slice(0, tokenized.length);
  }

  // Add an extra activation to illustrate the padding inputs in the UI.
  // This is added for illustration.
  const displayActivations =
    tf.tidy(() => activations.concat(tf.ones([1, EMBEDDING_DIM])));
  const displayActicationsArr = await displayActivations.array();

  tf.dispose([activations, prediction, displayActivations]);

  return {
    tokenized: tokenized,
    tokenScores: predsArr,
    tokenEmbeddings: displayActicationsArr,
  };
}

/**
 * Render the tokens
 *
 * @param {string[]} tokens the tokens
 * @param {Array.number[]} tokenScores model scores for each token
 * @param {Array.number[]} tokenEmbeddings token embeddings
 * @param {string} model name of model
 */
async function displayTokenization(tokens, tokenScores, tokenEmbeddings, model) {
  const resultsDiv = document.createElement('div');
  resultsDiv.classList = `tagging`;
  resultsDiv.innerHTML = `<p class="model-type ${model}">${model}</p>`;

  displayTokens(tokens, resultsDiv);
  displayEmbeddingsPlot(tokenEmbeddings, resultsDiv);
  displayTags(tokenScores, resultsDiv, model);

  document.getElementById('taggings').prepend(resultsDiv);
}

/**
 * Render the tokens.
 *
 * @param {string[]} tokens tokens to display
 * @param {HTMLElement} parentEl parent element
 */
function displayTokens(tokens, parentEl) {
  const tokensDiv = document.createElement('div');
  tokensDiv.classList = `tokens`;
  tokensDiv.innerHTML =
    tokens.map(token => `<div class="token">${token}</div>`).join('\n');
  parentEl.appendChild(tokensDiv);
}

const embeddingCol =
  d3.scaleSequential(d3.interpolateSpectral).domain([-0.075, 0.075]);
embeddingCol.clamp(true);

/**
 * Display an illustrative representation of the embeddings values
 * @param {*} embeddings
 * @param {*} parentEl
 */
function displayEmbeddingsPlot(embeddings, parentEl) {
  const embeddingDiv = document.createElement('div');
  embeddingDiv.classList = `embeddings`;

  embeddingDiv.innerHTML =
    embeddings
    .map(embedding => {
      // Note that this slice is arbitraty as the plot is only meant to
      // be illustrative.
      const embeddingValDivs = embedding.slice(0, 340).map(val => {
        return `<div class="embVal" ` +
          `style="background-color:${embeddingCol(val)} "` +
          `title="${val}"` +
          `></div>`;
      });

      return `<div class="embedding">` +
        `${embeddingValDivs.join('\n')}</div>`;
    })
    .join('\n');

  parentEl.appendChild(embeddingDiv);
}

/**
 *
 * @param {*} tokenScores
 * @param {*} parentEl
 * @param {*} modelName
 */
async function displayTags(tokenScores, parentEl, modelName) {
  const metadata = await loadMetadata(modelName);
  const {
    labels
  } = metadata;
  let location = "";

  const tagsDiv = document.createElement('div');
  tagsDiv.classList = `tags`;
  tagsDiv.innerHTML =
    tokenScores
    .map((scores, index) => {
      const maxIndex = scores.indexOf(Math.max(...scores));
      const token = labels[maxIndex];
      const tokenScore = (scores[maxIndex] * 100).toPrecision(3);
      return `<div class="tag ${token}">` +
        `&nbsp;&nbsp;${token.replace(/__/g, '')}<sup>` +
        `${tokenScore}%</sup></div>`;
    })
    .join('\n');

  if (location != "") {
    appendMessage(location, 'bot', messageId - 1);
  }
  parentEl.appendChild(tagsDiv);
}

async function tagMessage(inputText, model) {
  if (inputText != null && inputText.length > 0) {
    const result = await tagTokens(inputText, model);
    const {
      tokenized,
      tokenScores,
      tokenEmbeddings
    } = result;
    const metadata = await loadMetadata(model);
    const {
      labels
    } = metadata;
    const location = tokenScores
      .map((scores, index) => {
        const maxIndex = scores.indexOf(Math.max(...scores));
        if (maxIndex === 1) {
          return tokenized[index]
        }
      })
      .join('\ ')

    displayTokenization(tokenized, tokenScores, tokenEmbeddings, model);

    return location;
  }
}


async function getWeatherSearch(location) {
  const response = await fetch(`https://www.metaweather.com/api/location/search/?query=${location.trim()}`);

  const weatherSearch = response.json();


  return weatherSearch;
}

async function getWeather(location) {
  const weatherSearch = await getWeatherSearch(location);

  if (weatherSearch.length > 0) {
    const weatherResponse = await fetch(`https://www.metaweather.com/api/location/${weatherSearch[0].woeid}/`);
    const weather = await weatherResponse.json()

    return `The ${weather.location_type} of ${weather.title} is expecting ${weather.consolidated_weather[0].weather_state_name} today.`
  } else {
    return `I'm not smart enough to know weather data for ${location}`
  }
}


function speak(message) {
  let utterance = new SpeechSynthesisUtterance();

  utterance.text = message;
  utterance.rate = 5;
  utterance.pitch = 2;
  utterance.lang = "en-GB";

  speechSynthesis.speak(utterance);
}

function ask() {
  var SpeechRecognition = SpeechRecognition || webkitSpeechRecognition;
  var recognition = new SpeechRecognition();

  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.start();

  recognition.addEventListener('result', (e) => {
    let last = e.results.length - 1;
    let text = e.results[last][0].transcript;
    console.log(text);

    sendMessage(text);

    recognition.stop();
  });
}

let messageId = 0;

function appendMessage(message, sender, appendAfter) {
  const messageDiv = document.createElement('div');
  messageDiv.classList = `message ${sender}`;
  messageDiv.innerHTML = message;
  messageDiv.dataset.messageId = messageId++;

  const messageArea = document.getElementById('message-area');
  if (appendAfter == null) {
    messageArea.appendChild(messageDiv);
  } else {
    const inputMsg =
      document.querySelector(`.message[data-message-id="${appendAfter}"]`);
    inputMsg.parentNode.insertBefore(messageDiv, inputMsg.nextElementSibling);
  }

  // Scroll the message area to the bottom.
  messageArea.scroll({
    top: messageArea.scrollHeight,
    behavior: 'smooth'
  });

  // Return this message id so that a reply can be posted to it later
  return messageDiv.dataset.messageId;
}

async function sendMessage(inputText) {
  if (inputText != null && inputText.length > 0) {
    // Add the input text to the chat window
    const msgId = appendMessage(inputText, 'input');
    // Classify the text
    const classification = await classify([inputText]);
    // Add the response to the chat window
    const response = await getClassificationMessage(classification, inputText);
    appendMessage(response, 'bot', msgId);
  }
}

function setupListeners() {
  const form = document.getElementById('textentry');
  const textbox = document.getElementById('textbox');
  const speech = document.getElementById('speech');

  speech.addEventListener('click', ask, false);

  form.addEventListener('submit', event => {
    event.preventDefault();
    event.stopPropagation();

    const inputText = textbox.value;

    sendMessage(inputText);

    textbox.value = '';
  }, false);
}

window.addEventListener('load', function() {
  loadTaggerModel();
  setupListeners();
});
