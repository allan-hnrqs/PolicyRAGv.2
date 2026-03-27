(function () {
  const state = {
    chats: [createNewChat()],
    currentChatId: "chat-new",
    isLoading: false,
    nextChatSequence: 1
  };
  const MAX_CONTEXT_MESSAGES = 8;
  const MAX_CONTEXT_MESSAGE_CHARS = 1200;

  const elements = {
    hero: document.getElementById("heroState"),
    messages: document.getElementById("messages"),
    historyList: document.getElementById("historyList"),
    composer: document.getElementById("composer"),
    input: document.getElementById("messageInput"),
    newChatButton: document.getElementById("newChatButton")
  };

  function createNewChat() {
    return {
      id: "chat-new",
      title: "New Chat",
      messages: []
    };
  }

  function getCurrentChat() {
    return state.chats.find(function (chat) {
      return chat.id === state.currentChatId;
    }) || state.chats[0];
  }

  function slugify(text) {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 40);
  }

  function trimTitle(text) {
    return text.length > 28 ? text.slice(0, 28).trimEnd() + "..." : text;
  }

  function nextChatId(firstQuestion) {
    const nextSequence = state.nextChatSequence;
    state.nextChatSequence += 1;
    return "chat-" + nextSequence + "-" + slugify(firstQuestion || "policy-thread");
  }

  function normalizeLinks(citations) {
    const seen = new Set();
    return (citations || []).reduce(function (links, citation) {
      const url = citation && citation.canonical_url ? citation.canonical_url : "";
      if (!url || seen.has(url)) {
        return links;
      }
      seen.add(url);
      links.push({
        url: url,
        label: readableLink(url)
      });
      return links;
    }, []);
  }

  function splitAnswer(answerText) {
    return String(answerText || "")
      .split(/\n\s*\n/)
      .map(function (paragraph) {
        return paragraph.trim();
      })
      .filter(Boolean);
  }

  function toFriendlyMessage(message) {
    const text = String(message || "");

    if (text.indexOf("COHERE_API_KEY") >= 0) {
      return "PolicyAI could not reach its language model configuration. Check the repo .env file and try again.";
    }
    if (text.indexOf("Elasticsearch") >= 0) {
      return "PolicyAI could not reach the local search service. Start Elasticsearch and try again.";
    }
    if (
      text.indexOf("active index") >= 0 ||
      text.indexOf("build-index") >= 0 ||
      text.indexOf("Index manifest") >= 0 ||
      text.indexOf("embedding") >= 0
    ) {
      return "PolicyAI is not fully indexed yet. Build the active baseline index and try the question again.";
    }
    if (text) {
      return text;
    }
    return "PolicyAI could not answer that right now. Try again in a moment.";
  }

  function replaceNewChatIfNeeded(firstQuestion) {
    const current = getCurrentChat();
    if (current.id !== "chat-new" || current.messages.length > 0) {
      return current;
    }

    const nextChat = {
      id: nextChatId(firstQuestion),
      title: trimTitle(firstQuestion || "Policy Question"),
      messages: []
    };

    state.chats = [nextChat].concat(state.chats.filter(function (chat) {
      return chat.id !== "chat-new";
    }));
    state.chats.push(createNewChat());
    state.currentChatId = nextChat.id;
    return nextChat;
  }

  function renderHistory() {
    elements.historyList.innerHTML = "";

    const filledChats = state.chats.filter(function (chat) {
      return chat.messages.length > 0;
    });

    if (!filledChats.length) {
      const empty = document.createElement("div");
      empty.className = "history-item";
      empty.textContent = "No chats yet";
      empty.setAttribute("aria-disabled", "true");
      elements.historyList.appendChild(empty);
      return;
    }

    filledChats.forEach(function (chat) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "history-item" + (state.currentChatId === chat.id ? " active" : "");
      button.textContent = chat.title;
      button.addEventListener("click", function () {
        state.currentChatId = chat.id;
        render();
      });
      elements.historyList.appendChild(button);
    });
  }

  function renderMessages() {
    const current = getCurrentChat();
    const hasMessages = current.messages.length > 0;
    elements.hero.classList.toggle("hidden", hasMessages);
    elements.messages.innerHTML = "";

    current.messages.forEach(function (message) {
      const group = document.createElement("div");
      group.className = "message-group";

      if (message.role === "user") {
        group.innerHTML = [
          '<p class="message-label user">' + escapeHtml(message.label || "Buyer") + "</p>",
          '<div class="message-bubble user">' + escapeHtml(message.text) + "</div>"
        ].join("");
        elements.messages.appendChild(group);
        return;
      }

      if (message.pending) {
        group.innerHTML = [
          '<div class="assistant-head"><span class="bot-mark"></span><span>PolicyAI</span></div>',
          '<article class="assistant-card typing-card"><div class="typing-dots"><span></span><span></span><span></span></div></article>'
        ].join("");
        elements.messages.appendChild(group);
        return;
      }

      const paragraphs = (message.paragraphs || []).map(function (paragraph) {
        return "<p>" + escapeHtml(paragraph) + "</p>";
      }).join("");

      const sourceLinks = (message.sourceLinks || []).map(function (link) {
        return '<a class="source-link-pill" href="' + escapeAttribute(link.url) + '" target="_blank" rel="noreferrer noopener">' + escapeHtml(link.label) + "</a>";
      }).join("");

      let footerHtml = "";
      if (message.isError) {
        footerHtml = [
          '<footer class="assistant-footer">',
          '<div class="source-meta">',
          '<span class="source-dot" aria-hidden="true"></span>',
          "<div>",
          '<span class="source-label">' + escapeHtml(message.sourceLabel || "Status") + "</span>",
          '<span class="source-link">' + escapeHtml(message.sourceText || "Try again shortly.") + "</span>",
          "</div>",
          "</div>",
          "</footer>"
        ].join("");
      } else if (message.sourceLinks && message.sourceLinks.length) {
        footerHtml = [
          '<footer class="assistant-footer">',
          '<div class="source-meta">',
          '<span class="source-dot" aria-hidden="true"></span>',
          "<div>",
          '<span class="source-label">' + escapeHtml(message.sourceLabel || "Official sources") + "</span>",
          '<div class="source-links">' + sourceLinks + "</div>",
          "</div>",
          "</div>",
          '<div class="card-actions">',
          '<a class="icon-button" href="' + escapeAttribute(message.sourceLinks[0].url) + '" target="_blank" rel="noreferrer noopener" aria-label="Open source"><span class="icon-open"></span></a>',
          "</div>",
          "</footer>"
        ].join("");
      }

      group.innerHTML = [
        '<div class="assistant-head"><span class="bot-mark"></span><span>' + escapeHtml(message.label || "PolicyAI") + "</span></div>",
        '<article class="assistant-card' + (message.isError ? " error" : "") + '">',
        '<div class="assistant-body">' + paragraphs + "</div>",
        footerHtml,
        "</article>"
      ].join("");

      elements.messages.appendChild(group);
    });

    requestAnimationFrame(function () {
      window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
    });
  }

  function renderComposerState() {
    elements.input.disabled = state.isLoading;
    elements.composer.querySelector(".send-button").disabled = state.isLoading;
  }

  function render() {
    renderHistory();
    renderMessages();
    renderComposerState();
  }

  function removePendingAssistant(chat) {
    chat.messages = chat.messages.filter(function (message) {
      return !message.pending;
    });
  }

  function addErrorMessage(chat, text) {
    chat.messages.push({
      role: "assistant",
      label: "PolicyAI",
      isError: true,
      paragraphs: [toFriendlyMessage(text)],
      sourceLabel: "Live backend",
      sourceText: "Try again once the local backend services are available."
    });
  }

  function serializeConversation(chat) {
    return chat.messages
      .filter(function (message) {
        return !message.pending && !message.isError;
      })
      .map(function (message) {
        if (message.role === "user") {
          return {
            role: "user",
            content: String(message.text || "").slice(0, MAX_CONTEXT_MESSAGE_CHARS)
          };
        }
        return {
          role: "assistant",
          content: (message.paragraphs || []).join("\n\n").slice(0, MAX_CONTEXT_MESSAGE_CHARS)
        };
      })
      .filter(function (message) {
        return message.content.trim().length > 0;
      })
      .slice(-MAX_CONTEXT_MESSAGES);
  }

  async function submitQuestion(question) {
    const cleanQuestion = String(question || "").trim();
    if (!cleanQuestion || state.isLoading) {
      return;
    }

    const chat = replaceNewChatIfNeeded(cleanQuestion);
    chat.title = trimTitle(cleanQuestion);
    chat.messages.push({ role: "user", text: cleanQuestion, label: "Buyer" });
    chat.messages.push({ role: "assistant", pending: true });
    state.isLoading = true;
    elements.input.value = "";
    render();

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json"
        },
        body: JSON.stringify({
          question: cleanQuestion,
          messages: serializeConversation(chat)
        })
      });
      const payload = await response.json();
      removePendingAssistant(chat);

      if (!response.ok) {
        addErrorMessage(chat, payload.status_message || payload.error || "Backend query failed.");
      } else {
        chat.messages.push({
          role: "assistant",
          label: "PolicyAI",
          paragraphs: splitAnswer(payload.answer_text),
          sourceLabel: payload.response_mode === "rag" ? "Official sources" : "",
          sourceLinks: payload.response_mode === "rag" ? normalizeLinks(payload.citations) : [],
          responseMode: payload.response_mode || "general_guidance"
        });
      }
    } catch (error) {
      removePendingAssistant(chat);
      addErrorMessage(chat, "Could not reach the live backend server.");
    } finally {
      state.isLoading = false;
      render();
    }
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function escapeAttribute(text) {
    return escapeHtml(text).replace(/`/g, "&#96;");
  }

  function readableLink(url) {
    return String(url || "").replace(/^https?:\/\//, "");
  }

  elements.composer.addEventListener("submit", function (event) {
    event.preventDefault();
    submitQuestion(elements.input.value);
  });

  elements.newChatButton.addEventListener("click", function () {
    const existingEmpty = state.chats.find(function (chat) {
      return chat.id === "chat-new";
    });

    if (!existingEmpty) {
      state.chats.push(createNewChat());
    } else {
      existingEmpty.messages = [];
      existingEmpty.title = "New Chat";
    }

    state.currentChatId = "chat-new";
    render();
    elements.input.focus();
  });

  render();
})();
