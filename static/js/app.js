
const chat = document.getElementById('chat');
const form = document.getElementById('chatForm');
const question = document.getElementById('question');
const trace = document.getElementById('trace');
const sourceUsed = document.getElementById('sourceUsed');

let isProcessing = false;


// --------------------------------------------------
// HTML HELPERS
// --------------------------------------------------

function escapeHtml(s = '') {
    return String(s).replace(/[&<>'"]/g, c => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        "'": '&#39;',
        '"': '&quot;'
    }[c]));
}


function formatText(s = '') {
    return escapeHtml(s).replace(/\n/g, '<br>');
}


// --------------------------------------------------
// API RESPONSE HANDLER
// --------------------------------------------------

async function readResponse(response) {
    const text = await response.text();

    if (!text) {
        throw new Error(
            `Empty server response (HTTP ${response.status})`
        );
    }

    let data;

    try {
        data = JSON.parse(text);
    } catch {
        throw new Error(
            `Server returned invalid JSON (HTTP ${response.status})`
        );
    }

    if (!response.ok) {
        throw new Error(
            data.detail ||
            data.error ||
            `Request failed (${response.status})`
        );
    }

    return data;
}


// --------------------------------------------------
// MESSAGE RENDERING
// --------------------------------------------------

function addMessage(role, text, source = '', citations = []) {
    const wrap = document.createElement('div');

    wrap.className = `message ${role}`;

    const citeHtml = citations.length
        ? `
            <div class="citations">
                <strong>Sources</strong><br>
                ${citations.map(c => {
                    if (c.url) {
                        return `
                            <a href="${escapeHtml(c.url)}"
                               target="_blank"
                               rel="noopener noreferrer">
                               ${escapeHtml(c.title)}
                            </a>
                        `;
                    }

                    return escapeHtml(c.title);
                }).join('<br>')}
            </div>
        `
        : '';

    wrap.innerHTML = `
        <div class="avatar">${role === 'user' ? 'You' : 'AI'}</div>

        <div class="bubble">
            ${formatText(text)}

            ${source ? `
                <div class="answer-source">
                    Source: ${escapeHtml(source)}
                </div>
            ` : ''}

            ${citeHtml}
        </div>
    `;

    chat.appendChild(wrap);
    chat.scrollTop = chat.scrollHeight;
}


// --------------------------------------------------
// TRACE RENDERING
// --------------------------------------------------

function renderTrace(items = []) {
    trace.innerHTML = items.length
        ? items.map(item => `
            <div class="trace-item">
                ${escapeHtml(item)}
            </div>
        `).join('')
        : '<div class="empty">No trace.</div>';
}


// --------------------------------------------------
// LOADING STATE
// --------------------------------------------------

function setLoading(loading) {
    isProcessing = loading;

    const btn = form.querySelector('button');

    if (btn) {
        btn.disabled = loading;
        btn.textContent = loading ? 'Thinking...' : 'Send';
    }

    question.disabled = loading;

    document.querySelectorAll('.example').forEach(button => {
        button.disabled = loading;
    });

    sourceUsed.textContent = loading ? 'Processing...' : sourceUsed.textContent;
}


// --------------------------------------------------
// CHAT API
// --------------------------------------------------

async function askAgent(q) {

    if (isProcessing || !q.trim()) {
        return;
    }

    addMessage('user', q);

    question.value = '';

    renderTrace(['Starting LangGraph workflow...']);

    setLoading(true);

    try {

        const res = await fetch('/api/chat', {
            method: 'POST',

            headers: {
                'Content-Type': 'application/json'
            },

            body: JSON.stringify({
                question: q
            })
        });

        const data = await readResponse(res);

        addMessage(
            'assistant',
            data.answer || 'No answer was returned.',
            data.source_used || 'Unknown',
            data.citations || []
        );

        renderTrace(data.trace || []);

        sourceUsed.textContent = data.source_used || 'Unknown';

        if (
            data.rewritten_query &&
            data.rewritten_query !== q
        ) {
            renderTrace([
                ...(data.trace || []),
                `Final search query → ${data.rewritten_query}`
            ]);
        }

    } catch (error) {

        console.error('Chat API error:', error);

        addMessage(
            'assistant',
            `Error: ${error.message}`
        );

        renderTrace([
            'Request failed',
            error.message
        ]);

        sourceUsed.textContent = 'Error';

    } finally {

        setLoading(false);

    }
}


// --------------------------------------------------
// CHAT FORM
// --------------------------------------------------

form.addEventListener('submit', event => {

    event.preventDefault();

    const q = question.value.trim();

    if (q) {
        askAgent(q);
    }

});


// --------------------------------------------------
// EXAMPLE QUESTIONS
// --------------------------------------------------

document.querySelectorAll('.example').forEach(button => {

    button.addEventListener('click', () => {

        askAgent(button.textContent.trim());

    });

});


// --------------------------------------------------
// UPLOAD MODAL
// --------------------------------------------------

const modal = document.getElementById('uploadModal');

const openUpload = document.getElementById('openUpload');
const closeUpload = document.getElementById('closeUpload');

if (openUpload && modal) {
    openUpload.onclick = () => {
        modal.classList.remove('hidden');
    };
}

if (closeUpload && modal) {
    closeUpload.onclick = () => {
        modal.classList.add('hidden');
    };
}


// --------------------------------------------------
// DOCUMENT UPLOAD
// --------------------------------------------------

const uploadButton = document.getElementById('uploadBtn');

if (uploadButton) {

    uploadButton.onclick = async () => {

        const file = document.getElementById('fileInput').files[0];

        const key = document.getElementById('adminKey').value;

        const status = document.getElementById('uploadStatus');

        if (!file) {
            status.textContent = 'Choose a file first.';
            return;
        }

        uploadButton.disabled = true;

        status.textContent = 'Uploading and indexing document...';

        const fd = new FormData();

        fd.append('file', file);

        try {

            const response = await fetch('/api/ingest', {
                method: 'POST',

                headers: {
                    'X-Admin-Key': key
                },

                body: fd
            });

            const data = await readResponse(response);

            status.textContent =
                `Successfully indexed ${data.file}: ${data.chunks} chunks.`;

        } catch (error) {

            console.error('Upload API error:', error);

            status.textContent = `Error: ${error.message}`;

        } finally {

            uploadButton.disabled = false;

        }

    };

}
