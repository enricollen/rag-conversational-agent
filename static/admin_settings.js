// admin_settings.js

function updateApiKeyField() {
    const llmModelType = document.getElementById('llm_model_type').value;
    const openaiApiKeyField = document.getElementById('openai_api_key_field');
    const claudeApiKeyField = document.getElementById('claude_api_key_field');
    const embeddingModelName = document.getElementById('embedding_model_name');

    if (llmModelType === 'gpt') {
        openaiApiKeyField.style.display = 'block';
        claudeApiKeyField.style.display = 'none';
        updateEmbeddingModelNames('openai');
    } else if (llmModelType === 'claude') {
        openaiApiKeyField.style.display = 'none';
        claudeApiKeyField.style.display = 'block';
        updateEmbeddingModelNames('ollama');
    } else if (llmModelType === 'ollama') {
        openaiApiKeyField.style.display = 'none';
        claudeApiKeyField.style.display = 'none';
        updateEmbeddingModelNames('ollama');
    }

    updateLlmModelNames();
}

function updateLlmModelNames() {
    const llmModelType = document.getElementById('llm_model_type').value;
    const llmModelName = document.getElementById('llm_model_name');
    const otherLlmModelName = document.getElementById('llm_model_name_other');
    
    // Clear current options
    llmModelName.innerHTML = '';

    let options = [];

    if (llmModelType === 'gpt') {
        options = [
            { text: 'GPT 3.5', value: 'gpt-3.5-turbo' },
            { text: 'GPT-4o', value: 'gpt-4o' },
            { text: 'GPT-4', value: 'gpt-4' }
        ];
    } else if (llmModelType === 'ollama') {
        options = [
            { text: 'Llama3', value: 'llama3:8b' },
            { text: 'Gemma 2', value: 'gemma2' },
            { text: 'Mistral', value: 'mistral:7b' },
            { text: 'Other', value: 'other' }
        ];
    } else if (llmModelType === 'claude') {
        options = [
            { text: 'Claude 3.5 Sonnet', value: 'claude-3-5-sonnet-20240620' },
            { text: 'Claude 3 Opus', value: 'claude-3-opus-20240229' },
            { text: 'Claude 3 Sonnet', value: 'claude-3-sonnet-20240229' },
            { text: 'Claude 3 Haiku', value: 'claude-3-haiku-20240307' }
        ];
    }

    options.forEach(option => {
        const opt = document.createElement('option');
        opt.value = option.value;
        opt.textContent = option.text;
        llmModelName.appendChild(opt);
    });

    // Show or hide the "Other" input field based on the initial value
    if (llmModelName.value === 'other') {
        otherLlmModelName.style.display = 'block';
    } else {
        otherLlmModelName.style.display = 'none';
    }

    // Add event listener to show/hide "Other" input field
    llmModelName.addEventListener('change', function() {
        if (llmModelName.value === 'other') {
            otherLlmModelName.style.display = 'block';
        } else {
            otherLlmModelName.style.display = 'none';
        }
    });
}

function updateEmbeddingModelNames(selectedValue) {
    const embeddingModelName = document.getElementById('embedding_model_name');
    
    // Clear current options
    embeddingModelName.innerHTML = '';

    const option = document.createElement('option');
    option.value = selectedValue;
    option.textContent = selectedValue.charAt(0).toUpperCase() + selectedValue.slice(1);
    embeddingModelName.appendChild(option);

    // Automatically select the option
    embeddingModelName.value = selectedValue;
}

function handleFormSubmission(event) {
    const llmModelName = document.getElementById('llm_model_name');
    const llmModelNameOther = document.getElementById('llm_model_name_other');
    const embeddingModelName = document.getElementById('embedding_model_name');

    // If "Other" is selected, set the value to the text input value
    if (llmModelName.value === 'other') {
        llmModelName.value = llmModelNameOther.value;
    }

    // Enable embedding_model_name so its value can be submitted
    embeddingModelName.disabled = false;
}

window.onload = function() {
    updateApiKeyField();
    updateLlmModelNames();
    updateEmbeddingModelNames('openai'); // Default to OpenAI on load
};
