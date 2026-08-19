document.addEventListener('DOMContentLoaded', () => {
  const chatStream = document.getElementById('chatStream');
  const emptyState = document.getElementById('emptyState');
  const chatForm = document.getElementById('chatForm');
  const userInput = document.getElementById('userInput');
  const btnAction = document.getElementById('btnAction');
  const iconSend = btnAction.querySelector('.icon-send');
  const iconStop = btnAction.querySelector('.icon-stop');
  const btnNewChat = document.getElementById('btnNewChat');

  let isGenerating = false;
  let abortController = null;

  userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 200) + 'px';

    const hasText = userInput.value.trim().length > 0;
    if (!isGenerating) {
      if (hasText) {
        btnAction.classList.add('active');
        btnAction.removeAttribute('disabled');
      } else {
        btnAction.classList.remove('active');
        btnAction.setAttribute('disabled', 'true');
      }
    }
  });

  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isGenerating && userInput.value.trim().length > 0) {
        chatForm.dispatchEvent(new Event('submit'));
      }
    }
  });

  function renderMarkdown(text) {
    let html = text;
    html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

    html = html.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
      const language = lang || 'code';
      return `<div class="code-container">
        <div class="code-header">
          <span>${language}</span>
          <button class="btn-copy" onclick="copyCode(this)">Copy</button>
        </div>
        <pre><code>${code.trim()}</code></pre>
      </div>`;
    });

    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^\*]+)\*/g, '<em>$1</em>');
    html = html.replace(/\n/g, '<br>');

    return html;
  }

  window.copyCode = (btn) => {
    const code = btn.closest('.code-container').querySelector('code').textContent;
    navigator.clipboard.writeText(code).then(() => {
      btn.textContent = 'Copied!';
      setTimeout(() => btn.textContent = 'Copy', 2000);
    });
  };

  function appendMessage(role, text) {
    emptyState?.classList.add('hidden');

    const row = document.createElement('div');
    row.className = `message-row ${role}`;

    if (role === 'user') {
      const bubble = document.createElement('div');
      bubble.className = 'message-bubble';
      bubble.textContent = text;
      row.appendChild(bubble);
      chatStream.appendChild(row);
      chatStream.scrollTop = chatStream.scrollHeight;
      return bubble;
    } else {
      const body = document.createElement('div');
      body.className = 'message-body';
      body.innerHTML = renderMarkdown(text);
      row.appendChild(body);
      chatStream.appendChild(row);
      chatStream.scrollTop = chatStream.scrollHeight;
      return body;
    }
  }

  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (isGenerating) {
      if (abortController) abortController.abort();
      try {
        await fetch('/api/stop', { method: 'POST' });
      } catch (err) {}
      return;
    }

    const prompt = userInput.value.trim();
    if (!prompt) return;

    userInput.value = '';
    userInput.style.height = 'auto';
    btnAction.classList.remove('active');
    btnAction.setAttribute('disabled', 'true');

    appendMessage('user', prompt);

    const assistantBody = appendMessage('assistant', '');
    assistantBody.innerHTML = '<span class="cursor"></span>';

    isGenerating = true;
    btnAction.classList.add('generating');
    btnAction.removeAttribute('disabled');
    iconSend.classList.add('hidden');
    iconStop.classList.remove('hidden');

    abortController = new AbortController();
    let fullText = '';

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: abortController.signal,
        body: JSON.stringify({
          prompt: prompt,
          temp: 0.0,
          top_p: 0.9,
          top_k: 40,
          n_predict: 256,
          threads: 12
        })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = jsonParse(line.substring(6));
              if (data.type === 'token') {
                fullText += data.content;
                assistantBody.innerHTML = renderMarkdown(fullText) + '<span class="cursor"></span>';
                chatStream.scrollTop = chatStream.scrollHeight;
              } else if (data.type === 'done') {
                assistantBody.innerHTML = renderMarkdown(fullText);
              }
            } catch (err) {}
          }
        }
      }
      assistantBody.innerHTML = renderMarkdown(fullText);
    } catch (err) {
      if (err.name !== 'AbortError') {
        assistantBody.innerHTML = renderMarkdown(fullText) + '<br><span style="color:#ef4444;font-size:13px;">[Generation error]</span>';
      } else {
        assistantBody.innerHTML = renderMarkdown(fullText);
      }
    } finally {
      isGenerating = false;
      btnAction.classList.remove('generating');
      iconStop.classList.add('hidden');
      iconSend.classList.remove('hidden');
      if (userInput.value.trim().length === 0) {
        btnAction.classList.remove('active');
        btnAction.setAttribute('disabled', 'true');
      }
      userInput.focus();
    }
  });

  function jsonParse(str) {
    try {
      return JSON.parse(str);
    } catch (e) {
      return {};
    }
  }

  btnNewChat?.addEventListener('click', () => {
    if (isGenerating && abortController) {
      abortController.abort();
      fetch('/api/stop', { method: 'POST' }).catch(() => {});
    }
    chatStream.innerHTML = `
      <div class="empty-state" id="emptyState">
        <h1>How can I help you today?</h1>
      </div>
    `;
    userInput.value = '';
    userInput.style.height = 'auto';
    btnAction.classList.remove('active', 'generating');
    btnAction.setAttribute('disabled', 'true');
    iconStop.classList.add('hidden');
    iconSend.classList.remove('hidden');
    isGenerating = false;
    userInput.focus();
  });
});
