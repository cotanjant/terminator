use std::env;
use std::io::{self, BufRead, BufReader};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
    MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, List, ListItem, Paragraph, Wrap};
use ratatui::Terminal;
use serde::{Deserialize, Serialize};

const DEFAULT_API_URL: &str = "http://127.0.0.1:8000";

const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/help", "Show available slash commands"),
    ("/clear", "Clear conversation history"),
    ("/health", "Check API server connectivity (/v1/models)"),
    ("/url", "Show or switch API URL (usage: /url <http://...>)"),
    ("/exit", "Exit the CLI"),
];

#[derive(Clone, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
}

#[derive(Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    terminator_event: Option<TerminatorEvent>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
}

#[derive(Deserialize)]
struct StreamDelta {
    content: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum TerminatorEvent {
    #[serde(rename = "tool_call")]
    ToolCall {
        name: Option<String>,
        args: Option<serde_json::Value>,
    },
    #[serde(rename = "tool_output")]
    ToolOutput {
        #[allow(dead_code)]
        name: Option<String>,
        content: Option<String>,
        ok: Option<bool>,
        exit_code: Option<i64>,
    },
}

enum StreamMsg {
    Chunk(String),
    ToolCall {
        name: String,
        preview: String,
        cmd_full: String,
    },
    ToolOutput {
        content: String,
        ok: bool,
        exit_code: i64,
    },
    Done,
    Error(String),
}

enum DisplayItem {
    User(String),
    Assistant {
        raw: String,
        rendered: Vec<Line<'static>>,
    },
    Tool {
        name: String,
        preview: String,
        cmd_full: String,
        output_full: String,
        ok: bool,
        exit_code: i64,
        running: bool,
    },
    Info(String),
    Error(String),
}

struct App {
    api_url: String,
    connected_model: Option<String>,
    agent: ureq::Agent,
    messages: Vec<ChatMessage>,
    items: Vec<DisplayItem>,
    input: String,
    cursor: usize, // byte offset (always on char boundary)
    history: Vec<String>,
    history_idx: Option<usize>,
    expanded: bool,
    scroll_from_bottom: u16,
    busy: bool,
    rx: Option<Receiver<StreamMsg>>,
    spinner_idx: usize,
    slash_idx: usize,
}

impl App {
    fn new(api_url: String) -> Self {
        let agent = ureq::AgentBuilder::new().build();
        let connected_model = check_health(&agent, &api_url).ok();
        Self {
            api_url,
            connected_model,
            agent,
            messages: Vec::new(),
            items: Vec::new(),
            input: String::new(),
            cursor: 0,
            history: Vec::new(),
            history_idx: None,
            expanded: false,
            scroll_from_bottom: 0,
            busy: false,
            rx: None,
            spinner_idx: 0,
            slash_idx: 0,
        }
    }

    fn slash_matches(&self) -> Vec<(&'static str, &'static str)> {
        if !self.input.starts_with('/') || self.input.contains(' ') {
            return Vec::new();
        }
        SLASH_COMMANDS
            .iter()
            .copied()
            .filter(|(cmd, _)| cmd.starts_with(&self.input))
            .collect()
    }

    fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor, c);
        self.cursor += c.len_utf8();
        self.slash_idx = 0;
    }

    fn backspace(&mut self) {
        if self.cursor > 0 {
            let prev = self.input[..self.cursor]
                .char_indices()
                .last()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.input.drain(prev..self.cursor);
            self.cursor = prev;
            self.slash_idx = 0;
        }
    }

    fn delete_char(&mut self) {
        if self.cursor < self.input.len() {
            let next = self.input[self.cursor..]
                .char_indices()
                .nth(1)
                .map(|(i, _)| self.cursor + i)
                .unwrap_or(self.input.len());
            self.input.drain(self.cursor..next);
            self.slash_idx = 0;
        }
    }

    fn move_left(&mut self) {
        if self.cursor > 0 {
            self.cursor = self.input[..self.cursor]
                .char_indices()
                .last()
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
    }

    fn move_right(&mut self) {
        if self.cursor < self.input.len() {
            self.cursor = self.input[self.cursor..]
                .char_indices()
                .nth(1)
                .map(|(i, _)| self.cursor + i)
                .unwrap_or(self.input.len());
        }
    }

    fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let next = match self.history_idx {
            None => self.history.len() - 1,
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.history_idx = Some(next);
        self.input = self.history[next].clone();
        self.cursor = self.input.len();
    }

    fn history_down(&mut self) {
        let Some(i) = self.history_idx else {
            return;
        };
        if i + 1 < self.history.len() {
            let next = i + 1;
            self.history_idx = Some(next);
            self.input = self.history[next].clone();
        } else {
            self.history_idx = None;
            self.input.clear();
        }
        self.cursor = self.input.len();
    }

    fn submit(&mut self) -> bool {
        let line = self.input.trim().to_string();
        self.input.clear();
        self.cursor = 0;
        self.history_idx = None;
        self.slash_idx = 0;

        if line.is_empty() {
            return true;
        }
        self.history.push(line.clone());
        self.scroll_from_bottom = 0;

        if line == "exit" || line == "quit" {
            return false;
        }

        if line.starts_with('/') {
            return self.handle_slash(&line);
        }

        if self.busy {
            self.items
                .push(DisplayItem::Info("Wait for the current response to finish.".into()));
            return true;
        }

        self.items.push(DisplayItem::User(line.clone()));
        self.messages.push(ChatMessage {
            role: "user".to_string(),
            content: line,
        });

        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);
        self.busy = true;

        let agent = self.agent.clone();
        let api_url = self.api_url.clone();
        let msgs = self.messages.clone();

        thread::spawn(move || {
            stream_worker(agent, api_url, msgs, tx);
        });

        true
    }

    fn handle_slash(&mut self, line: &str) -> bool {
        let mut parts = line.splitn(2, char::is_whitespace);
        let cmd = parts.next().unwrap_or("").to_lowercase();
        let arg = parts.next().unwrap_or("").trim();

        match cmd.as_str() {
            "/exit" => return false,
            "/help" => {
                let mut help = String::from("Slash Commands:\n");
                for (c, d) in SLASH_COMMANDS {
                    help.push_str(&format!("  {c:<10} {d}\n"));
                }
                self.items.push(DisplayItem::Info(help.trim_end().to_string()));
            }
            "/clear" => {
                self.messages.clear();
                self.items.clear();
                self.items
                    .push(DisplayItem::Info("Conversation cleared.".into()));
            }
            "/health" => match check_health(&self.agent, &self.api_url) {
                Ok(model) => {
                    self.connected_model = Some(model.clone());
                    self.items.push(DisplayItem::Info(format!(
                        "API OK ({}, model: {model})",
                        self.api_url
                    )));
                }
                Err(err) => {
                    self.connected_model = None;
                    self.items
                        .push(DisplayItem::Error(format!("API unreachable: {err}")));
                }
            },
            "/url" => {
                if arg.is_empty() {
                    self.items
                        .push(DisplayItem::Info(format!("API URL: {}", self.api_url)));
                } else {
                    self.api_url = arg.to_string();
                    self.connected_model = check_health(&self.agent, &self.api_url).ok();
                    self.items.push(DisplayItem::Info(format!(
                        "Switched API URL to {}",
                        self.api_url
                    )));
                }
            }
            _ => {
                self.items.push(DisplayItem::Error(format!(
                    "Unknown command: {cmd}. Type /help for commands."
                )));
            }
        }
        true
    }

    fn poll_stream(&mut self) -> bool {
        let Some(rx) = self.rx.take() else {
            return false;
        };
        let mut changed = false;
        let mut disconnected = false;
        loop {
            match rx.try_recv() {
                Ok(StreamMsg::Chunk(tok)) => {
                    if let Some(DisplayItem::Assistant { raw, rendered }) = self.items.last_mut() {
                        raw.push_str(&tok);
                        *rendered = render_markdown_lines(raw);
                    } else {
                        let rendered = render_markdown_lines(&tok);
                        self.items.push(DisplayItem::Assistant { raw: tok, rendered });
                    }
                    self.scroll_from_bottom = 0;
                    changed = true;
                }
                Ok(StreamMsg::ToolCall {
                    name,
                    preview,
                    cmd_full,
                }) => {
                    self.items.push(DisplayItem::Tool {
                        name,
                        preview,
                        cmd_full,
                        output_full: String::new(),
                        ok: true,
                        exit_code: 0,
                        running: true,
                    });
                    self.scroll_from_bottom = 0;
                    changed = true;
                }
                Ok(StreamMsg::ToolOutput {
                    content,
                    ok,
                    exit_code,
                }) => {
                    for item in self.items.iter_mut().rev() {
                        if let DisplayItem::Tool {
                            output_full,
                            ok: item_ok,
                            exit_code: item_code,
                            running,
                            ..
                        } = item
                        {
                            if *running {
                                *output_full = content;
                                *item_ok = ok;
                                *item_code = exit_code;
                                *running = false;
                                break;
                            }
                        }
                    }
                    self.scroll_from_bottom = 0;
                    changed = true;
                }
                Ok(StreamMsg::Done) => {
                    if let Some(DisplayItem::Assistant { raw, .. }) = self.items.last() {
                        self.messages.push(ChatMessage {
                            role: "assistant".to_string(),
                            content: raw.clone(),
                        });
                    }
                    self.busy = false;
                    disconnected = true;
                    changed = true;
                    break;
                }
                Ok(StreamMsg::Error(err)) => {
                    self.messages.pop();
                    self.items.push(DisplayItem::Error(err));
                    self.busy = false;
                    disconnected = true;
                    changed = true;
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.busy = false;
                    disconnected = true;
                    changed = true;
                    break;
                }
            }
        }
        if !disconnected {
            self.rx = Some(rx);
        }
        changed
    }
}

fn check_health(agent: &ureq::Agent, api_url: &str) -> Result<String, String> {
    let url = format!("{}/v1/models", api_url.trim_end_matches('/'));
    let resp = agent
        .get(&url)
        .timeout(Duration::from_secs(3))
        .call()
        .map_err(|e| format!("{e}"))?;
    let body: serde_json::Value = resp.into_json().map_err(|e| format!("{e}"))?;
    Ok(body["data"][0]["id"]
        .as_str()
        .unwrap_or("terminator")
        .to_string())
}

fn summarize_cmd(cmd: &str) -> String {
    let first_line = cmd
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .unwrap_or("");
    let chars: Vec<char> = first_line.chars().collect();
    if chars.len() > 48 {
        let short: String = chars[..45].iter().collect();
        format!("{short}...")
    } else {
        first_line.to_string()
    }
}

fn stream_worker(
    agent: ureq::Agent,
    api_url: String,
    messages: Vec<ChatMessage>,
    tx: mpsc::Sender<StreamMsg>,
) {
    let url = format!("{}/v1/chat/completions", api_url.trim_end_matches('/'));
    let req = ChatRequest {
        model: "terminator",
        messages: &messages,
        stream: true,
    };

    let body_json = match serde_json::to_value(&req) {
        Ok(v) => v,
        Err(e) => {
            let _ = tx.send(StreamMsg::Error(format!("{e}")));
            return;
        }
    };

    let resp = match agent
        .post(&url)
        .timeout(Duration::from_secs(120))
        .send_json(body_json)
    {
        Ok(r) => r,
        Err(e) => {
            let _ = tx.send(StreamMsg::Error(format!("{e}")));
            return;
        }
    };

    let reader = BufReader::new(resp.into_reader());
    for line_res in reader.lines() {
        let line = match line_res {
            Ok(l) => l,
            Err(e) => {
                let _ = tx.send(StreamMsg::Error(format!("{e}")));
                return;
            }
        };
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) {
            if let Some(event) = chunk.terminator_event {
                match event {
                    TerminatorEvent::ToolCall { name, args } => {
                        let tool_name = name.unwrap_or_else(|| "tool".to_string());
                        let cmd_full = if tool_name == "run_command" {
                            args.as_ref()
                                .and_then(|a| a.get("cmd"))
                                .and_then(|c| c.as_str())
                                .unwrap_or("")
                                .to_string()
                        } else {
                            args.map(|a| a.to_string()).unwrap_or_default()
                        };
                        let preview = summarize_cmd(&cmd_full);
                        let _ = tx.send(StreamMsg::ToolCall {
                            name: tool_name,
                            preview,
                            cmd_full,
                        });
                    }
                    TerminatorEvent::ToolOutput {
                        content,
                        ok,
                        exit_code,
                        ..
                    } => {
                        let exit_code = exit_code.unwrap_or(0);
                        let ok = ok.unwrap_or(exit_code == 0);
                        let _ = tx.send(StreamMsg::ToolOutput {
                            content: content.unwrap_or_default(),
                            ok,
                            exit_code,
                        });
                    }
                }
            }
            if let Some(choice) = chunk.choices.first() {
                if let Some(tok) = &choice.delta.content {
                    if !tok.is_empty() {
                        let _ = tx.send(StreamMsg::Chunk(tok.clone()));
                    }
                }
            }
        }
    }

    let _ = tx.send(StreamMsg::Done);
}

fn parse_inline_spans(line: &str) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let chars: Vec<char> = line.chars().collect();
    let mut buf = String::new();
    let mut i = 0;

    while i < chars.len() {
        if i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == '*' {
            if let Some(end) = (i + 2..chars.len().saturating_sub(1))
                .find(|&j| chars[j] == '*' && chars[j + 1] == '*')
            {
                if !buf.is_empty() {
                    spans.push(Span::raw(std::mem::take(&mut buf)));
                }
                let bold_text: String = chars[i + 2..end].iter().collect();
                spans.push(Span::styled(
                    bold_text,
                    Style::default().add_modifier(Modifier::BOLD),
                ));
                i = end + 2;
                continue;
            }
        }
        if chars[i] == '`' {
            if let Some(end) = (i + 1..chars.len()).find(|&j| chars[j] == '`') {
                if !buf.is_empty() {
                    spans.push(Span::raw(std::mem::take(&mut buf)));
                }
                let code_text: String = chars[i + 1..end].iter().collect();
                spans.push(Span::styled(
                    code_text,
                    Style::default().fg(Color::Cyan),
                ));
                i = end + 1;
                continue;
            }
        }
        buf.push(chars[i]);
        i += 1;
    }

    if !buf.is_empty() {
        spans.push(Span::raw(buf));
    }
    if spans.is_empty() {
        spans.push(Span::raw(String::new()));
    }
    spans
}

fn render_markdown_lines(text: &str) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut in_code = false;

    for raw in text.lines() {
        let trimmed = raw.trim_start();
        if trimmed.starts_with("```") {
            in_code = !in_code;
            let lang = trimmed.trim_start_matches('`').trim();
            if in_code && !lang.is_empty() {
                lines.push(Line::from(Span::styled(
                    format!("  {lang}"),
                    Style::default().fg(Color::DarkGray),
                )));
            }
            continue;
        }

        if in_code {
            lines.push(Line::from(Span::styled(
                format!("  {raw}"),
                Style::default().fg(Color::Cyan),
            )));
            continue;
        }

        if trimmed == "---" || trimmed == "***" {
            lines.push(Line::from(Span::styled(
                "────────────────────────────────────────",
                Style::default().fg(Color::DarkGray),
            )));
            continue;
        }

        if let Some(rest) = trimmed
            .strip_prefix("### ")
            .or_else(|| trimmed.strip_prefix("## "))
            .or_else(|| trimmed.strip_prefix("# "))
        {
            lines.push(Line::from(Span::styled(
                rest.to_string(),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )));
            continue;
        }

        if let Some(rest) = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
        {
            let indent = raw.len() - trimmed.len();
            let mut spans = vec![
                Span::raw(" ".repeat(indent)),
                Span::styled("• ", Style::default().fg(Color::Green)),
            ];
            spans.extend(parse_inline_spans(rest));
            lines.push(Line::from(spans));
            continue;
        }

        lines.push(Line::from(parse_inline_spans(raw)));
    }
    lines
}

fn build_chat_lines(app: &App) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let spin_frames = ["○", "◔", "◑", "◕"];
    let braille_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

    if app.items.is_empty() {
        let status = match &app.connected_model {
            Some(m) => Span::styled(
                format!("Connected ({m} @ {})", app.api_url),
                Style::default().fg(Color::Green),
            ),
            None => Span::styled(
                format!("Disconnected ({}) — start: .venv/bin/uvicorn api:app --port 8000", app.api_url),
                Style::default().fg(Color::Yellow),
            ),
        };
        out.push(Line::from(vec![
            Span::styled(
                "Terminator ",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            status,
        ]));
        out.push(Line::from(Span::styled(
            "Type / for commands, Ctrl+O to expand/collapse tools, PgUp/PgDn to scroll.",
            Style::default().fg(Color::DarkGray),
        )));
        out.push(Line::from(""));
    }

    for item in &app.items {
        match item {
            DisplayItem::User(msg) => {
                out.push(Line::from(""));
                out.push(Line::from(vec![
                    Span::styled(
                        "❯ ",
                        Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        msg.clone(),
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                ]));
            }
            DisplayItem::Tool {
                name,
                preview,
                cmd_full,
                output_full,
                ok,
                exit_code,
                running,
            } => {
                if *running {
                    let frame = spin_frames[app.spinner_idx % spin_frames.len()];
                    out.push(Line::from(vec![
                        Span::styled(format!("{frame} "), Style::default().fg(Color::Yellow)),
                        Span::styled(
                            name.clone(),
                            Style::default().add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            format!("({preview})"),
                            Style::default().fg(Color::DarkGray),
                        ),
                    ]));
                } else {
                    let dot_color = if *ok { Color::Green } else { Color::Red };
                    let hint = if app.expanded {
                        "  (press Ctrl+O to collapse)"
                    } else {
                        "  (press Ctrl+O to expand)"
                    };
                    let mut spans = vec![
                        Span::styled("● ", Style::default().fg(dot_color)),
                        Span::styled(
                            name.clone(),
                            Style::default().add_modifier(Modifier::BOLD),
                        ),
                    ];
                    if !app.expanded {
                        spans.push(Span::styled(
                            format!("({preview})"),
                            Style::default().fg(Color::DarkGray),
                        ));
                    }
                    if !*ok {
                        spans.push(Span::styled(
                            format!(" (exit {exit_code})"),
                            Style::default().fg(Color::Red),
                        ));
                    }
                    spans.push(Span::styled(hint, Style::default().fg(Color::DarkGray)));
                    out.push(Line::from(spans));

                    if app.expanded {
                        for line in cmd_full.trim_end_matches('\n').lines() {
                            out.push(Line::from(Span::styled(
                                format!("  {line}"),
                                Style::default().fg(Color::White),
                            )));
                        }
                        let trimmed_out = output_full.trim_end_matches('\n');
                        if trimmed_out.is_empty() {
                            out.push(Line::from(Span::styled(
                                "  └ (no output)",
                                Style::default().fg(Color::DarkGray),
                            )));
                        } else {
                            out.push(Line::from(Span::styled(
                                "  └ output:",
                                Style::default().fg(Color::DarkGray),
                            )));
                            for line in trimmed_out.lines() {
                                out.push(Line::from(Span::styled(
                                    format!("    {line}"),
                                    Style::default().fg(Color::DarkGray),
                                )));
                            }
                        }
                    }
                }
            }
            DisplayItem::Assistant { rendered, .. } => {
                out.push(Line::from(""));
                out.extend(rendered.iter().cloned());
            }
            DisplayItem::Info(msg) => {
                for l in msg.lines() {
                    out.push(Line::from(Span::styled(
                        l.to_string(),
                        Style::default().fg(Color::Cyan),
                    )));
                }
            }
            DisplayItem::Error(err) => {
                out.push(Line::from(Span::styled(
                    format!("Error: {err}"),
                    Style::default().fg(Color::Red),
                )));
            }
        }
    }

    let waiting_on_llm = app.busy
        && !matches!(
            app.items.last(),
            Some(DisplayItem::Tool { running: true, .. })
        );
    if waiting_on_llm {
        let frame = braille_frames[app.spinner_idx % braille_frames.len()];
        out.push(Line::from(Span::styled(
            format!("{frame} Thinking..."),
            Style::default().fg(Color::Cyan),
        )));
    }

    out
}

fn wrapped_line_count(lines: &[Line<'_>], width: u16) -> u16 {
    let w = (width as usize).max(1);
    lines
        .iter()
        .map(|l| l.width().max(1).div_ceil(w) as u16)
        .sum()
}

fn ui(f: &mut ratatui::Frame, app: &App) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area);

    let chat_lines = build_chat_lines(app);
    let total_lines = wrapped_line_count(&chat_lines, chunks[0].width);
    let chat_para = Paragraph::new(chat_lines).wrap(Wrap { trim: false });
    let visible = chunks[0].height;
    let max_scroll = total_lines.saturating_sub(visible);
    let clamped_from_bottom = app.scroll_from_bottom.min(max_scroll);
    let scroll_y = max_scroll.saturating_sub(clamped_from_bottom);
    f.render_widget(chat_para.scroll((scroll_y, 0)), chunks[0]);

    let sep = "─".repeat(area.width as usize);
    f.render_widget(
        Paragraph::new(Span::styled(sep, Style::default().fg(Color::DarkGray))),
        chunks[1],
    );

    let prompt_line = Line::from(vec![
        Span::styled(
            "❯ ",
            Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
        ),
        Span::raw(app.input.clone()),
    ]);
    f.render_widget(Paragraph::new(prompt_line), chunks[2]);

    let mode_label = if app.expanded { "expanded" } else { "compact" };
    let status_text = format!(
        " Ctrl+O: tools ({mode_label}) · PgUp/PgDn: scroll · /: commands · Ctrl+D: quit"
    );
    f.render_widget(
        Paragraph::new(Span::styled(status_text, Style::default().fg(Color::DarkGray))),
        chunks[3],
    );

    let matches = app.slash_matches();
    if !matches.is_empty() {
        let h = (matches.len() as u16).min(8);
        let popup_y = chunks[2].y.saturating_sub(h);
        let popup_area = Rect::new(0, popup_y, area.width.min(72), h);
        f.render_widget(Clear, popup_area);
        let list_items: Vec<ListItem> = matches
            .iter()
            .enumerate()
            .map(|(i, (cmd, desc))| {
                let style = if i == app.slash_idx {
                    Style::default().bg(Color::DarkGray).fg(Color::White)
                } else {
                    Style::default()
                };
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!(" {cmd:<10} "),
                        style.fg(Color::Cyan).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(*desc, style.fg(Color::Gray)),
                ]))
            })
            .collect();
        f.render_widget(List::new(list_items), popup_area);
    }

    let cursor_cols = app.input[..app.cursor].chars().count() as u16;
    f.set_cursor_position((chunks[2].x + 2 + cursor_cols, chunks[2].y));
}

struct TermGuard;

impl Drop for TermGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let mut out = io::stdout();
        let _ = execute!(out, LeaveAlternateScreen, DisableMouseCapture);
    }
}

fn main() -> io::Result<()> {
    let api_url = env::var("TERMINATOR_API_URL").unwrap_or_else(|_| DEFAULT_API_URL.to_string());
    let mut app = App::new(api_url);

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let _guard = TermGuard;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut last_tick = Instant::now();
    let mut dirty = true;
    'main: loop {
        if app.poll_stream() {
            dirty = true;
        }

        if dirty {
            terminal.draw(|f| ui(f, &app))?;
            dirty = false;
        }

        let timeout = if app.busy {
            Duration::from_millis(30)
        } else {
            Duration::from_millis(200)
        };

        if event::poll(timeout)? {
            loop {
                match event::read()? {
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        dirty = true;
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                        match key.code {
                            KeyCode::Char('c') if ctrl => break 'main,
                            KeyCode::Char('d') if ctrl => break 'main,
                            KeyCode::Char('o') if ctrl => {
                                app.expanded = !app.expanded;
                            }
                            KeyCode::PageUp => {
                                app.scroll_from_bottom = app.scroll_from_bottom.saturating_add(5);
                            }
                            KeyCode::PageDown => {
                                app.scroll_from_bottom = app.scroll_from_bottom.saturating_sub(5);
                            }
                            KeyCode::Tab => {
                                let matches = app.slash_matches();
                                if let Some((cmd, _)) = matches.get(app.slash_idx) {
                                    app.input = format!("{cmd} ");
                                    app.cursor = app.input.len();
                                }
                            }
                            KeyCode::Up => {
                                let matches = app.slash_matches();
                                if !matches.is_empty() {
                                    app.slash_idx = app.slash_idx.saturating_sub(1);
                                } else {
                                    app.history_up();
                                }
                            }
                            KeyCode::Down => {
                                let matches = app.slash_matches();
                                if !matches.is_empty() {
                                    if app.slash_idx + 1 < matches.len() {
                                        app.slash_idx += 1;
                                    }
                                } else {
                                    app.history_down();
                                }
                            }
                            KeyCode::Left => app.move_left(),
                            KeyCode::Right => app.move_right(),
                            KeyCode::Home => app.cursor = 0,
                            KeyCode::End => app.cursor = app.input.len(),
                            KeyCode::Backspace => app.backspace(),
                            KeyCode::Delete => app.delete_char(),
                            KeyCode::Enter => {
                                let matches = app.slash_matches();
                                if !matches.is_empty() {
                                    if let Some((cmd, _)) = matches.get(app.slash_idx) {
                                        app.input = (*cmd).to_string();
                                        app.cursor = app.input.len();
                                    }
                                }
                                if !app.submit() {
                                    break 'main;
                                }
                            }
                            KeyCode::Char(c) if !ctrl => app.insert_char(c),
                            _ => {}
                        }
                    }
                    Event::Mouse(mouse) => match mouse.kind {
                        MouseEventKind::ScrollUp => {
                            app.scroll_from_bottom = app.scroll_from_bottom.saturating_add(3);
                            dirty = true;
                        }
                        MouseEventKind::ScrollDown => {
                            app.scroll_from_bottom = app.scroll_from_bottom.saturating_sub(3);
                            dirty = true;
                        }
                        _ => {}
                    },
                    Event::Resize(_, _) => {
                        dirty = true;
                    }
                    _ => {}
                }
                if !event::poll(Duration::ZERO)? {
                    break;
                }
            }
        }

        if app.busy && last_tick.elapsed() >= Duration::from_millis(80) {
            app.spinner_idx = app.spinner_idx.wrapping_add(1);
            last_tick = Instant::now();
            dirty = true;
        }
    }

    Ok(())
}
