# SAGE Flash Mode — Operator Manual

This manual explains how to configure and operate Flash Mode without needing to know whether inference is physically running on the laptop, a friend's RTX 4060, Kaggle, or a private company GPU server.

## 1. What Flash Mode is

Flash Mode has three fixed model roles:

| Role | Purpose | Foreground/background |
|---|---|---|
| Gemma | First-pass reasoning, routing, text answers, synthesis | Foreground |
| Qwen | Image understanding, OCR, and visual extraction | Foreground, only when Gemma requests it |
| Memory 2B | Compresses completed turns into durable conversation memory | Background |

The SAGE backend always runs the orchestration. Each model role can be bound to local hardware or an OpenAI-compatible remote endpoint. The chat UI does not need to know whether that endpoint is Kaggle, an RTX 4060 reached through Cloudflare, or a private server.

```text
Browser
  -> local SAGE backend
       -> Gemma role  -> local GPU or remote endpoint
       -> Qwen role   -> local GPU or remote endpoint
       -> Memory role -> disabled, local CPU, shared remote, or separate remote
```

## 2. Starting SAGE

From the SAGE directory:

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:8899
```

Do not open `static/index.html` directly from the filesystem. The UI requires the FastAPI backend.

## 3. Opening the Flash Runtime panel

1. Click **Settings** at the bottom-left of the sidebar.
2. In the **Model** section, click **Flash runtime**.
3. The full-screen **Inference control center** opens.

Direct emergency/deep link:

```text
http://127.0.0.1:8899/#flash-runtime
```

The small Settings panel contains ordinary UI preferences. Hardware and endpoint configuration is intentionally placed in the larger modal so it is not confined to the narrow sidebar.

## 4. One-time local configuration

Copy `.env.example` to `.env` and set at least:

```dotenv
LLAMA_SERVER_PATH=C:\path\to\llama-server.exe
MODEL_DIR=C:\path\to\models
SAGE_FLASH_DEFAULT=1
```

Flash local ports default to:

| Role | Port |
|---|---:|
| Gemma | 8090 |
| Qwen | 8091 |
| Memory | 8092 |

The editable model catalog is `flash_models.json`. For every role, verify:

- `repo`: Hugging Face repository used by remote deployment.
- `file`: exact GGUF filename, including capitalization.
- `mmproj`: exact multimodal projector filename for Qwen, otherwise `null`.
- `context`, `temperature`, and `max_tokens`.
- `estimated_vram_gib`: used by local and bridge VRAM safety checks.
- `reasoning`: llama.cpp reasoning setting.

For local execution, `file` and `mmproj` must already exist inside `MODEL_DIR`. For bridge deployment, the bridge downloads them from `repo`.

The memory role is configured for `unsloth/Qwen3.5-2B-MTP-GGUF` using
`Qwen3.5-2B-IQ4_NL.gguf`. The Qwen vision projector filename is
`mmproj-F16.gguf`.

## 5. Configuration recipes

### Case 1 — Laptop has RTX 4060 8 GB

Use this when SAGE runs on the GPU laptop.

1. Under **Gemma + Qwen inference**, choose **Local RTX GPU**.
2. Choose one memory option:
   - **Disabled** while testing the core pair.
   - **Local CPU / RAM** to run the 2B model through local llama.cpp with GPU layers disabled.
   - **Separate remote server** to send only memory work to Kaggle or another host.
3. Click **Apply for this session**.
4. Click **Test connections**.

Testing starts the local models lazily. Gemma and Qwen remain on their separate ports and stay resident, avoiding the legacy model-switching loop.

Recommended initial setup:

```text
Gemma:  local_gpu
Qwen:   local_gpu
Memory: disabled
```

After Gemma and Qwen are confirmed healthy, enable the memory model.

### Case 2 — No local GPU, remote RTX 4060 available

1. Choose **Remote GPU server**.
2. Enter the Cloudflare/private server root URL, for example:

   ```text
   https://random-name.trycloudflare.com
   ```

   A URL ending in `/v1` is also accepted and normalized automatically.

3. Enter the bearer API key issued by the hosted bridge. A key can be omitted only when the server intentionally allows unauthenticated access, such as a trusted local-LAN endpoint.
4. Keep the default model IDs `gemma` and `qwen` unless the bridge instances use different IDs.
5. For memory, choose:
   - **Same remote server as Gemma + Qwen** if the memory instance is registered on that bridge.
   - **Separate remote server** if memory uses another Cloudflare URL, another computer, or Kaggle.
6. Click **Apply for this session**, then **Test connections**.

### Case 3 — Use Kaggle for all three models

1. Start the multi-model SAGE bridge in Kaggle and expose it through Cloudflare.
2. Choose **Remote GPU server** and enter the Kaggle tunnel URL/key.
3. Choose **Same remote server as Gemma + Qwen** for memory.
4. Ensure the bridge has instances named `gemma`, `qwen`, and `memory`, or change the three model-ID fields to match the actual instance IDs.
5. Apply and test.

The bridge may place the models on either T4:

```text
GPU #0: Gemma + Qwen
GPU #1: Memory
```

or any other arrangement that stays within its VRAM guard. SAGE routes by model ID, not GPU number.

Use three separate dictionaries in `STARTUP_MODELS`. Repeating `id`, `repo`,
`file`, and other keys inside one dictionary does not create three models;
Python silently keeps only the final value.

Correct Cell 1 model block:

```python
STARTUP_MODELS = [
    {
        "id": "memory",
        "repo": "unsloth/Qwen3.5-2B-MTP-GGUF",
        "file": "Qwen3.5-2B-IQ4_NL.gguf",
        "mmproj": None,
        "gpu": 0,
        "context": 8192,
        "reasoning": "off",
        "est_size_gib": 2.0,
        "token": None,
    },
    {
        "id": "qwen",
        "repo": "unsloth/Qwen3-VL-2B-Instruct-GGUF",
        "file": "Qwen3-VL-2B-Instruct-IQ4_NL.gguf",
        "mmproj": "mmproj-F16.gguf",
        "gpu": 0,
        "context": 4096,
        "reasoning": "off",
        "est_size_gib": 2.2,
        "token": None,
    },
    {
        "id": "gemma",
        "repo": "unsloth/gemma-4-E2B-it-GGUF",
        "file": "gemma-4-E2B-it-IQ4_NL.gguf",
        "mmproj": None,
        "gpu": 1,
        "context": 8192,
        "reasoning": "on",
        "est_size_gib": 3.6,
        "token": None,
    },
]
```

These are public repositories, so an HF token is normally unnecessary. If a
token is needed later, store it in a Kaggle secret rather than writing it into
the notebook. The Hugging Face token downloads model files; it is not the
bridge API key used by SAGE.

Cell 3 generates a fresh `API_KEY` and prints it beside `PANEL ENDPOINT`. Those
two printed values are what belong in the SAGE runtime panel.

### Mixed example — Local interactive pair, Kaggle memory

```text
Gemma + Qwen: Local RTX GPU
Memory:       Separate remote server
```

Enter only the Kaggle memory URL/key. Foreground chat stays local while completed turns are compressed remotely in the background.

## 6. Applying, testing, and deploying

### Apply for this session

**Apply for this session** sends the selected bindings to the local SAGE backend.

- The URL and key are held in process memory.
- They are not written to `.env`, JSON, local storage, or the model catalog.
- They disappear when the SAGE backend restarts.
- A browser refresh clears the form fields, but the backend binding remains active until the backend restarts or a new profile is applied.

### Test connections

Testing performs the following:

- Local role: validates files and llama-server, checks VRAM, starts the role server, and waits for health.
- Remote role: calls authenticated `GET /v1/models` and confirms the exact configured model ID is listed.
- Disabled memory: reports healthy/disabled.

Do not start chatting until Gemma and Qwen both show **ready**.

### Deploy model

Deployment is available only when at least one remote bridge is configured.

1. Apply a remote profile.
2. Select `Gemma`, `Qwen`, or `Memory 2B`.
3. Select the primary or memory endpoint.
4. Select GPU #0 or GPU #1.
5. Click **Deploy model**.
6. Repeat for the other roles.
7. Click **Test connections**.

SAGE sends the selected role's fixed `repo`, `file`, `mmproj`, context, reasoning setting, and estimated VRAM from `flash_models.json` to:

```text
POST /control/models/start
```

The bridge downloads and starts the model. A bridge VRAM rejection or llama-server load error is returned to the modal instead of being silently swallowed.

## 7. Remote-server contract

A generic company server does not need to be Kaggle-specific. It needs these OpenAI-compatible routes:

```text
GET  /v1/models
POST /v1/chat/completions
```

Requests use:

```http
Authorization: Bearer <api-key>
Content-Type: application/json
```

The `model` value is the role's configured model ID. This permits one URL to route requests to several independently loaded model instances.

Remote deployment is optional. To support deployment from SAGE, the server must additionally provide:

```text
POST /control/models/start
```

A company server that preloads its own models only needs the two `/v1` routes.

## 8. What happens when a message is sent

Gemma always runs first and chooses one Flash case:

| Case | Execution |
|---|---|
| A | Gemma answers directly. Qwen is not called. |
| B | Gemma delegates a visual request and Qwen's answer goes directly to the user. |
| C | Qwen extracts visual evidence, then Gemma performs final synthesis. |
| D | Gemma's completed text portion and Qwen's independent visual portion are combined. |

There is no separate router-model inference. Routing and Gemma's useful work happen in the same first call.

### Attachments

- Images are encoded and sent to Qwen only when Gemma delegates visual work.
- Plain-text formats are inserted directly into Gemma's shared request context.
- Complex documents are ingested through the existing SAGE document database and relevant chunks are retrieved for Gemma.
- Attachment metadata and registered document IDs remain compatible with the old architecture.

## 9. Background memory

If enabled, the 2B memory role receives the completed user/assistant turn after the answer is ready. It creates a concise durable summary containing useful facts, preferences, commitments, and unresolved tasks.

Important behavior:

- Memory work uses one bounded background worker.
- It does not delay the foreground answer.
- Recent summaries from the same chat session are added to later Gemma requests.
- Memory summaries are isolated by session ID.
- Clicking **Reset** clears summaries for the open session and creates a new session ID.

The Memory tab is not yet a full editor for these Flash summaries. The implemented role is currently the background compression path.

## 10. Chat-bar controls

### FLASH

Active and fully functional. Messages use `POST /api/flash`.

### REASONING · SOON

A deliberate placeholder. Clicking it shows **Reasoning Mode is coming soon** and leaves Flash active. It does not silently switch to the old architecture.

### Reset

The Reset button:

1. Clears visible messages.
2. Removes the active chat entry.
3. Deletes stored Flash summaries for that session.
4. Generates a fresh session ID.

It does not erase other conversations or their memories.

## 11. Privacy and network behavior

When every enabled role is local, the network card reports local/air-gapped execution.

When any role is remote, it changes to **Private Remote GPU Active** and displays only the configured endpoint hostnames. API keys are never returned by status APIs; status exposes only `has_api_key: true/false`.

Remote operation necessarily sends the relevant prompt and any required attachment data to the configured server. Whether that server is private and trustworthy is controlled by its operator.

## 12. Troubleshooting

### Settings does not open

- Hard-refresh the page with `Ctrl+F5` after updating SAGE.
- Confirm the page is served from `http://127.0.0.1:8899`, not opened as a local HTML file.
- Use `http://127.0.0.1:8899/#flash-runtime` to open the runtime modal directly.

### Local model file not found

Check that the exact `file` and `mmproj` names in `flash_models.json` exist in `MODEL_DIR`. Linux/Kaggle filenames are case-sensitive.

### llama-server is not configured

Set `LLAMA_SERVER_PATH` in `.env` to the actual executable.

### Insufficient free VRAM

- Close other GPU applications.
- Confirm the selected quant files match their estimated sizes.
- Disable local memory; use local CPU or a remote memory endpoint.
- Do not run the old legacy model server alongside the resident Flash pair.

### Remote test says model ID was not listed

Open the bridge's `/v1/models` response and copy its exact `id` values into the Gemma, Qwen, and memory model-ID fields.

The runtime test also shows every model ID reported by the bridge and
automatically adopts a unique obvious match such as `Gemma Flash Controller`.
Using the stable IDs `gemma`, `qwen`, and `memory` in the notebook remains the
simplest and least ambiguous setup.

### Only one startup model appears

Every model must be its own dictionary inside the list. A single dictionary
cannot contain the same keys three times. Python retains only the last `id`,
`repo`, `file`, and related values, so only the final model starts.

### HTTP 401 or 403

The URL is reachable but the API key is missing or incorrect. Restarting the tunnel may also mean the URL/key pair changed.

### HTTP 404

Use the server root URL. SAGE accepts a trailing `/v1`, but do not paste `/v1/chat/completions` as the endpoint.

### Deployment fails while loading a GGUF

Read the complete error shown in the control center and the bridge's model log. Common causes are:

- wrong quant filename or repository;
- incompatible/corrupt GGUF;
- wrong or mismatched multimodal projector;
- unsupported llama.cpp model architecture;
- VRAM safety guard rejection;
- model metadata tensor-shape errors.

Changing GPU number cannot fix an invalid GGUF tensor shape; use a compatible model export.

### Chat works but memory does not

- Confirm memory is not Disabled.
- Replace the placeholder memory catalog entry.
- Test the memory role and verify its exact remote model ID.
- Remember that memory runs after the answer, so its failure does not fail foreground chat.

## 13. Fast operator checklists

### Local RTX checklist

- `LLAMA_SERVER_PATH` valid.
- Gemma/Qwen files present in `MODEL_DIR`.
- Qwen `mmproj` present.
- Local RTX selected.
- Memory disabled for first test.
- Apply → Test → both foreground roles ready.

### Remote checklist

- Tunnel/server currently running.
- Root URL current.
- Bearer key current.
- `/v1/models` lists the expected IDs.
- UI model IDs match exactly.
- Apply → Test.

### Kaggle deployment checklist

- Bridge started with API secret.
- Cloudflare URL copied after the current restart.
- Correct repos/files in `flash_models.json`.
- Apply remote profile.
- Deploy each role to the intended GPU.
- Test all roles before sending a chat.
