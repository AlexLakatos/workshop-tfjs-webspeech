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

import * as bot from './bot';

import NexmoClient from 'nexmo-client';

var activeConversation;
fetch('http://localhost:3000/api/new')
  .then(function(response) {
    return response.json();
  })
  .then(function(response) {
    new NexmoClient({ debug: false })
        .login(response.jwt)
        .then(app => {
          console.log('*** Logged into app', app)
          return app.getConversation(response.conversation.id)
        })
        .then(conversation => {
          console.log('*** Retrieved conversations', conversation);
          activeConversation = conversation;
        })
        .catch(console.error)
  });

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
    const classification = await bot.classify([inputText]);
    // Add the response to the chat window
    const response = await bot.getClassificationMessage(classification, inputText);
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
  setupListeners();
});
