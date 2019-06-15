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
        response = '⛅'
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
