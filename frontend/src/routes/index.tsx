import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useEffect, useMemo, useState } from "react";
import { ArrowDown, ArrowRight, ArrowUpRight, BookOpen, Check, ChevronDown, Copy, Download, Github, Menu, Phone, Mail, Play, RotateCcw, Sparkles, X, Cpu, Server } from "lucide-react";
import { Button } from "@/components/ui/button";
import bookImage from "@/assets/shakespeare-still-life.jpg";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Rayvat Outsourcing — Generative AI LSTM Text Generator" },
      { name: "description", content: "Rayvat AI Labs presentation of Shakespeare Word-Level PyTorch LSTM Text Generation System." },
      { property: "og:title", content: "Rayvat Outsourcing — Generative AI LSTM Text Generator" },
      { property: "og:description", content: "Explore the live PyTorch LSTM text generator powered by Rayvat Outsourcing technology." },
      { property: "og:type", content: "website" },
    ],
  }),
  component: Index,
});

const SUGGESTIONS = ["to be or not", "the king", "my lord", "shall i compare", "first citizen"];
const architecture = [
  { index: "01", title: "Data Preprocessing", description: "Clean Gutenberg metadata, lowercase text, strip punctuation, normalize whitespace, and construct 30-word input sequences.", meta: "RAW TEXT → WORDS → ID SEQUENCES" },
  { index: "02", title: "PyTorch Stacked LSTM", description: "Word Embedding (128-dim) + 2 Stacked LSTM layers (256 hidden units, 0.2 dropout) + Dense Linear head mapping to 12,850 vocabulary logits.", meta: "EMBED → STACKED LSTM → LINEAR" },
  { index: "03", title: "Temperature Text Generation", description: "Autoregressively predict next word -> append -> slide sequence window using Temperature sampling (0.5, 0.8, 1.0).", meta: "SEED PROMPT → NEXT WORD LOOP" },
];

type Chain = Map<string, string[]>;
function buildChain(text: string): Chain {
  const words = text.toLowerCase().match(/[a-z]+/g) ?? [];
  const chain: Chain = new Map();
  for (let i = 0; i < words.length - 3; i++) {
    const key = words.slice(i, i + 3).join(" ");
    const next = chain.get(key) ?? [];
    const following = words[i + 3];
    if (!following) continue;
    next.push(following);
    chain.set(key, next);
  }
  return chain;
}

function makePreview(seed: string, chain: Chain, count: number, temperature: number): string[] {
  const words = seed.toLowerCase().match(/[a-z]+/g) ?? [];
  if (!words.length || chain.size === 0) return [];
  const keys = [...chain.keys()];
  const output = [...words];
  for (let i = 0; i < count; i++) {
    let key = output.slice(-3).join(" ");
    let possibilities = chain.get(key);
    if (!possibilities?.length) {
      const tail = output.slice(-2).join(" ");
      const matching = keys.filter(k => k.startsWith(`${tail} `));
      if (!matching.length) {
        const last = output.at(-1);
        if (last) matching.push(...keys.filter(k => k.split(" ")[1] === last));
      }
      key = matching[Math.floor(Math.random() * matching.length)] ?? keys[Math.floor(Math.random() * keys.length)] ?? "";
      possibilities = chain.get(key);
    }
    if (!possibilities?.length) break;
    const counts = new Map<string, number>();
    for (const word of possibilities) counts.set(word, (counts.get(word) ?? 0) + 1);
    const choices = [...counts.entries()].map(([word, frequency]) => ({ word, weight: Math.pow(frequency, 1 / temperature) }));
    const total = choices.reduce((sum, choice) => sum + choice.weight, 0);
    let draw = Math.random() * total;
    let selected = choices[0]?.word;
    for (const choice of choices) {
      draw -= choice.weight;
      if (draw <= 0) { selected = choice.word; break; }
    }
    if (!selected) break;
    output.push(selected);
  }
  return output;
}

function Index() {
  const [seed, setSeed] = useState("to be or not");
  const [length, setLength] = useState(44);
  const [temperature, setTemperature] = useState(0.8);
  const [corpus, setCorpus] = useState("");
  const [words, setWords] = useState<string[]>([]);
  const [visibleCount, setVisibleCount] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isBackendLive, setIsBackendLive] = useState(false);
  const [copied, setCopied] = useState(false);
  const [mobileMenu, setMobileMenu] = useState(false);

  const chain = useMemo(() => buildChain(corpus), [corpus]);

  useEffect(() => {
    fetch("/data/shakespeare-preview.txt").then(r => r.text()).then(setCorpus).catch(() => setCorpus(""));
    
    // Check if live PyTorch Flask server is online (Render or localhost)
    const checkHealth = async () => {
      try {
        const r1 = await fetch("https://rayvat-generative-ai-lstm-text-generator.onrender.com/api/health");
        if (r1.ok) { setIsBackendLive(true); return; }
      } catch (e) {}
      try {
        const r2 = await fetch("http://localhost:5000/api/health");
        if (r2.ok) setIsBackendLive(true);
      } catch (e) { setIsBackendLive(false); }
    };
    checkHealth();
  }, []);

  useEffect(() => {
    if (!isGenerating || visibleCount >= words.length) {
      if (isGenerating && visibleCount >= words.length) setIsGenerating(false);
      return;
    }
    const timer = window.setTimeout(() => setVisibleCount(n => Math.min(n + 1, words.length)), 40);
    return () => window.clearTimeout(timer);
  }, [visibleCount, words, isGenerating]);

  const generate = useCallback(async () => {
    if (!seed.trim()) return;
    setIsGenerating(true);
    setCopied(false);

    // Try calling live PyTorch Flask server first
    const endpoints = [
      "https://rayvat-generative-ai-lstm-text-generator.onrender.com/api/generate",
      "http://localhost:5000/api/generate"
    ];

    for (const url of endpoints) {
      try {
        const response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ seed: seed, length: length, temperature: temperature })
        });
        if (response.ok) {
          const data = await response.json();
          if (data.text) {
            const generatedWords = data.text.split(/\s+/);
            setWords(generatedWords);
            setVisibleCount(seed.split(/\s+/).length);
            setIsBackendLive(true);
            return;
          }
        }
      } catch (e) {}
    }

    // Fallback to instant chain preview
    const output = makePreview(seed, chain, length, temperature);
    if (!output.length) {
      setIsGenerating(false);
      return;
    }
    setWords(output);
    setVisibleCount(Math.min((seed.toLowerCase().match(/[a-z]+/g) ?? []).length, output.length));
  }, [seed, chain, length, temperature]);

  const copyText = async () => {
    if (!visibleCount) return;
    await navigator.clipboard.writeText(words.slice(0, visibleCount).join(" "));
    setCopied(true);
    window.setTimeout(() => setCopied(false), 2000);
  };

  const displayedText = words.slice(0, visibleCount).join(" ");

  return (
    <div className="min-h-screen bg-background text-foreground font-sans">
      {/* Rayvat Top Bar */}
      <div className="bg-[#002b49] text-white text-[11px] py-1.5 px-5 sm:px-8 lg:px-14 flex flex-wrap justify-between items-center border-b border-[#003d66]">
        <div className="font-mono tracking-[0.25em] text-[#ff7700] uppercase font-bold text-[10px] sm:text-[11px]">
          BELIEVE &nbsp;•••&nbsp; BEGIN &nbsp;•••&nbsp; BECOME
        </div>
        <div className="flex items-center gap-6 text-gray-300 text-[11px]">
          <span className="flex items-center gap-1.5"><Phone className="size-3 text-[#ff7700]"/> Toll Free: +91 8000322044</span>
          <span className="flex items-center gap-1.5 max-sm:hidden"><Mail className="size-3 text-[#ff7700]"/> hr@rayvat.com</span>
        </div>
      </div>

      {/* Main Rayvat Header */}
      <header className="sticky top-0 z-30 border-b border-line bg-background/95 backdrop-blur-xl">
        <div className="mx-auto flex h-[76px] max-w-[1480px] items-center justify-between px-5 sm:px-8 lg:px-14">
          <a href="#top" className="flex items-center gap-3" aria-label="Rayvat Outsourcing AI Portal">
            <div className="flex items-center justify-center font-extrabold text-2xl tracking-tighter text-[#004b87]">
              <span className="text-[#f36c21] font-black text-3xl mr-0.5">R</span>AYVAT
            </div>
            <div className="h-6 w-px bg-line" />
            <span className="font-display text-[12px] font-bold uppercase leading-[1.15] tracking-[.14em] text-foreground">
              OUTSOURCING<br/>
              <span className="text-[#f36c21]">AI LABS</span>
            </span>
          </a>

          <nav className="hidden items-center gap-8 lg:flex" aria-label="Main navigation">
            <a className="text-[12px] font-semibold uppercase tracking-[.12em] text-muted-foreground transition-colors hover:text-[#f36c21]" href="#playground">Generator</a>
            <a className="text-[12px] font-semibold uppercase tracking-[.12em] text-muted-foreground transition-colors hover:text-[#f36c21]" href="#how-it-works">Architecture</a>
            <a className="text-[12px] font-semibold uppercase tracking-[.12em] text-muted-foreground transition-colors hover:text-[#f36c21]" href="#project">Assignment Spec</a>
            <a className="text-[12px] font-semibold uppercase tracking-[.12em] text-muted-foreground transition-colors hover:text-[#f36c21]" href="https://rayvat.com" target="_blank" rel="noreferrer">Company Site</a>
          </nav>

          <div className="hidden items-center gap-4 lg:flex">
            <a href="tel:+918000322044" className="rounded-sm bg-[#004b87] px-4 py-2 text-[11px] font-bold uppercase tracking-[.09em] text-white hover:bg-[#003866] transition-colors flex items-center gap-1.5">
              <Phone className="size-3"/> Call HR
            </a>
            <a href="mailto:hr@rayvat.com" className="rounded-sm bg-[#f36c21] px-4 py-2 text-[11px] font-bold uppercase tracking-[.09em] text-white hover:bg-[#d85610] transition-colors">
              CAREER
            </a>
          </div>

          <Button variant="ghost" size="icon" className="lg:hidden" aria-label={mobileMenu ? "Close menu" : "Open menu"} onClick={() => setMobileMenu(!mobileMenu)}>
            {mobileMenu ? <X/> : <Menu/>}
          </Button>
        </div>

        {mobileMenu && (
          <nav className="flex flex-col gap-1 border-t border-line bg-background px-5 py-4 lg:hidden" aria-label="Mobile navigation">
            <a onClick={() => setMobileMenu(false)} href="#playground" className="py-2.5 text-sm font-medium text-foreground">Generator</a>
            <a onClick={() => setMobileMenu(false)} href="#how-it-works" className="py-2.5 text-sm font-medium text-foreground">Architecture</a>
            <a onClick={() => setMobileMenu(false)} href="#project" className="py-2.5 text-sm font-medium text-foreground">Assignment Spec</a>
            <a href="https://rayvat.com" target="_blank" rel="noreferrer" className="py-2.5 text-sm font-bold text-[#f36c21]">Rayvat Official Website</a>
          </nav>
        )}
      </header>

      <main id="top">
        {/* Hero Section */}
        <section className="relative overflow-hidden border-b border-line bg-gradient-to-b from-[#001d36]/10 to-transparent">
          <div className="mx-auto grid max-w-[1480px] grid-cols-1 gap-9 px-5 pb-11 pt-11 sm:px-8 lg:grid-cols-[1.04fr_.96fr] lg:gap-14 lg:px-14 lg:pb-16 lg:pt-16">
            <div className="relative z-10 flex flex-col justify-center animate-enter">
              <div className="mb-5 flex items-center gap-3 text-[11px] font-mono uppercase tracking-[.16em] text-[#f36c21]">
                <span className="h-px w-7 bg-[#f36c21]" /> RAYVAT OUTSOURCING • GENERATIVE AI PORTAL
              </div>
              <h1 className="max-w-[720px] font-literary text-[clamp(3.5rem,5.8vw,7rem)] font-medium leading-[.9] tracking-normal text-foreground">
                Shakespeare <em className="font-normal text-[#f36c21]">LSTM</em><br/>Text Generator<span className="text-[#f36c21]">.</span>
              </h1>
              <p className="mt-6 max-w-[520px] text-[15px] leading-[1.8] text-muted-foreground sm:text-[16px]">
                Official submission assignment developed for <strong>Rayvat Outsourcing</strong>. Powered by a word-level PyTorch LSTM neural network trained on Shakespeare's Complete Works with early stopping and temperature sampling.
              </p>
              
              <div className="mt-8 flex flex-wrap items-center gap-4">
                <Button asChild className="h-12 rounded-sm bg-[#f36c21] px-6 text-[12px] font-bold uppercase tracking-[.09em] text-white hover:bg-[#d85610] transition-transform hover:-translate-y-0.5 shadow-md">
                  <a href="#playground">Try Live Generator <ArrowRight className="ml-2 size-4"/></a>
                </Button>
                <a href="#how-it-works" className="group flex items-center gap-2 text-[12px] font-semibold uppercase tracking-[.09em] text-foreground hover:text-[#f36c21]">
                  Explore Architecture <ArrowDown className="size-4 transition-transform group-hover:translate-y-1" />
                </a>
              </div>

              <div className="mt-10 flex gap-7 border-t border-line pt-6 sm:gap-12">
                <div><span className="block font-literary text-[28px] leading-none text-[#004b87]">Word-Level</span><span className="mt-2 block font-mono text-[10px] uppercase tracking-[.12em] text-muted-foreground">TOKENIZATION</span></div>
                <div><span className="block font-literary text-[28px] leading-none text-[#004b87]">Stacked LSTM</span><span className="mt-2 block font-mono text-[10px] uppercase tracking-[.12em] text-muted-foreground">2 LAYERS (256-DIM)</span></div>
                <div><span className="block font-literary text-[28px] leading-none text-[#004b87]">PyTorch</span><span className="mt-2 block font-mono text-[10px] uppercase tracking-[.12em] text-muted-foreground">FRAMEWORK</span></div>
              </div>
            </div>

            <div className="relative min-h-[360px] overflow-hidden rounded-sm bg-surface sm:min-h-[440px] lg:min-h-[500px] border border-line shadow-lg">
              <img src={bookImage} width={1024} height={1024} alt="Shakespeare Still Life" className="absolute inset-0 h-full w-full object-cover" />
              <div className="absolute bottom-0 left-0 right-0 flex items-end justify-between bg-gradient-to-t from-[#001d36]/90 via-[#001d36]/50 to-transparent px-6 pb-6 pt-24 text-white sm:px-8">
                <div>
                  <span className="font-mono text-[10px] uppercase tracking-[.16em] text-[#f36c21]">RAYVAT AI LABS PORTAL</span>
                  <p className="mt-1 font-literary text-[26px] italic leading-none">Shakespeare's Complete Works</p>
                </div>
                <BookOpen className="size-6 text-[#f36c21]" strokeWidth={1.4}/>
              </div>
            </div>
          </div>
        </section>

        {/* Live Playground Section */}
        <section id="playground" className="scroll-mt-[76px] border-b border-line py-16 sm:py-22">
          <div className="mx-auto max-w-[1480px] px-5 sm:px-8 lg:px-14">
            <div className="mb-8 flex flex-wrap items-end justify-between gap-5">
              <div>
                <div className="mb-3 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[.17em] text-[#f36c21]">
                  <span className="size-1.5 rounded-full bg-[#f36c21]"/> INTERACTIVE DEMO / 01
                </div>
                <h2 className="font-literary text-[clamp(2.8rem,4.5vw,5rem)] leading-[.95] text-foreground">
                  Text Generator Playground<span className="text-[#f36c21]">.</span>
                </h2>
                <p className="mt-3 max-w-xl text-sm leading-relaxed text-muted-foreground">
                  Provide a word seed prompt to trigger the PyTorch LSTM next-word prediction engine.
                </p>
              </div>

              <div className="flex items-center gap-2.5 border border-line px-3.5 py-2 rounded-sm font-mono text-[10px] uppercase tracking-[.08em] bg-surface">
                <span className={`size-2 rounded-full ${isBackendLive ? "bg-emerald-500 animate-pulse" : "bg-amber-500"}`}/>
                <span className="text-foreground font-semibold">{isBackendLive ? "PYTORCH BACKEND LIVE (PORT 5000)" : "LOCAL ENGINE ACTIVE"}</span>
              </div>
            </div>

            <div className="grid overflow-hidden rounded-sm border border-line bg-surface lg:grid-cols-[.85fr_1.15fr] shadow-xl">
              {/* Left Controls */}
              <div className="border-b border-line p-6 sm:p-8 lg:border-b-0 lg:border-r lg:p-10">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[11px] uppercase tracking-[.16em] text-muted-foreground">01 / INPUT SEED PROMPT</span>
                  <Sparkles className="size-4 text-[#f36c21]" strokeWidth={1.6}/>
                </div>

                <label htmlFor="seed" className="mt-7 block font-literary text-[28px] leading-none">Enter Seed Words</label>
                <p className="mt-2 text-[12px] text-muted-foreground">Provide starting words for autoregressive next-word prediction.</p>
                
                <textarea 
                  id="seed" 
                  value={seed} 
                  maxLength={140} 
                  onChange={e => setSeed(e.target.value)} 
                  className="mt-4 min-h-[95px] w-full resize-none rounded-sm border border-line bg-background p-4 font-literary text-[22px] italic text-foreground outline-none placeholder:text-muted-foreground focus:border-[#f36c21]" 
                  placeholder="to be or not..." 
                />

                <div className="mt-3 flex flex-wrap gap-2">
                  {SUGGESTIONS.map(s => (
                    <Button 
                      key={s} 
                      variant="outline" 
                      size="sm" 
                      onClick={() => setSeed(s)} 
                      className={`h-7 rounded-sm border-line px-2.5 font-literary text-[14px] italic transition-colors ${seed === s ? "bg-[#004b87] text-white border-[#004b87]" : "bg-transparent text-muted-foreground hover:bg-surface-raised hover:text-foreground"}`}
                    >
                      {s}
                    </Button>
                  ))}
                </div>

                <div className="my-6 h-px bg-line" />

                <div className="flex items-center justify-between">
                  <label htmlFor="length" className="text-[12px] font-semibold text-foreground">Output Length (Words)</label>
                  <span className="font-mono text-[11px] font-bold text-[#f36c21]">{length} WORDS</span>
                </div>
                <input id="length" type="range" min="20" max="100" step="5" value={length} onChange={e => setLength(Number(e.target.value))} className="mt-3 w-full accent-[#f36c21]" />

                <div className="mt-5 flex items-center justify-between">
                  <label htmlFor="temperature" className="text-[12px] font-semibold text-foreground">Sampling Temperature (T)</label>
                  <span className="font-mono text-[11px] font-bold text-[#f36c21]">{temperature.toFixed(1)}</span>
                </div>
                <input id="temperature" type="range" min="0.3" max="1.5" step="0.1" value={temperature} onChange={e => setTemperature(Number(e.target.value))} className="mt-3 w-full accent-[#f36c21]" />
                <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">Lower T (0.5) is structured & deterministic; higher T (1.0) is diverse.</p>

                <Button onClick={generate} disabled={!seed.trim() || isGenerating} className="mt-6 h-12 w-full rounded-sm bg-[#f36c21] text-[12px] font-bold uppercase tracking-[.1em] text-white hover:bg-[#d85610] transition-colors">
                  <Play className="mr-2 size-3.5 fill-current"/>
                  {isGenerating ? "Predicting Next Words..." : "Generate Text Response"}
                  <ArrowRight className="ml-auto size-4" />
                </Button>
              </div>

              {/* Right Output */}
              <div className="flex min-h-[500px] flex-col bg-background/50 p-6 sm:p-8 lg:p-10">
                <div className="flex items-center justify-between gap-4">
                  <span className="font-mono text-[11px] uppercase tracking-[.16em] text-muted-foreground">02 / GENERATED WORD SEQUENCE</span>
                  <span className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[.1em] text-[#004b87]">
                    <span className={`size-2 rounded-full ${isGenerating ? "bg-[#f36c21] animate-ping" : "bg-[#004b87]"}`}/> 
                    {isGenerating ? "GENERATING" : "READY"}
                  </span>
                </div>

                <div className="mt-6 flex flex-1 flex-col justify-center border-y border-line py-8">
                  {visibleCount ? (
                    <div aria-live="polite" className={`max-w-[650px] font-literary text-[clamp(1.8rem,3vw,3.2rem)] leading-[1.28] text-foreground ${isGenerating ? "typing-caret" : ""}`}>
                      <span className="text-[#f36c21]">“</span>{displayedText}<span className="text-[#f36c21]">”</span>
                    </div>
                  ) : (
                    <div className="max-w-[520px]">
                      <span className="font-literary text-[80px] leading-none text-[#f36c21]/40">“</span>
                      <p className="mt-1 font-literary text-[clamp(1.8rem,2.8vw,2.6rem)] italic leading-[1.15] text-muted-foreground">
                        Click 'Generate Text Response' to generate text...
                      </p>
                      <p className="mt-5 font-mono text-[10px] uppercase tracking-[.12em] text-muted-foreground">
                        GENERATED TEXT WILL DISPLAY HERE
                      </p>
                    </div>
                  )}
                </div>

                <div className="mt-5 flex flex-wrap items-center justify-between gap-4">
                  <div className="font-mono text-[10px] uppercase tracking-[.12em] text-muted-foreground">
                    {visibleCount ? `${visibleCount} words · PyTorch Word-Level Output` : "Awaiting input prompt"}
                  </div>
                  <div className="flex items-center gap-2">
                    <Button variant="outline" size="icon" disabled={!visibleCount} onClick={copyText} title="Copy text" className="size-9 rounded-sm border-line bg-transparent hover:bg-surface-raised">
                      {copied ? <Check className="text-emerald-500"/> : <Copy/>}
                    </Button>
                    <Button variant="outline" size="icon" disabled={isGenerating} onClick={generate} title="Generate again" className="size-9 rounded-sm border-line bg-transparent hover:bg-surface-raised">
                      <RotateCcw/>
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Architecture Specs */}
        <section id="how-it-works" className="scroll-mt-[76px] border-b border-line bg-[#001d36] text-white py-16 sm:py-22">
          <div className="mx-auto max-w-[1480px] px-5 sm:px-8 lg:px-14">
            <div className="mb-12 grid gap-5 lg:grid-cols-[.9fr_1.1fr] lg:items-end">
              <div>
                <div className="mb-3 font-mono text-[10px] uppercase tracking-[.16em] text-[#f36c21]">SYSTEM ARCHITECTURE / 02</div>
                <h2 className="font-literary text-[clamp(3rem,4.8vw,5.5rem)] leading-[.9] text-white">
                  Technical <em className="font-normal text-[#f36c21]">Design.</em>
                </h2>
              </div>
              <p className="max-w-[490px] text-[15px] leading-[1.8] text-gray-300">
                Designed according to Rayvat Outsourcing engineering specifications: word tokenization, sequence pairing, stacked PyTorch LSTM layers, early stopping, and perplexity tracking.
              </p>
            </div>

            <div className="grid border-t border-white/20 md:grid-cols-3">
              {architecture.map((item, i) => (
                <article key={item.index} className={`flex min-h-[250px] flex-col border-b border-white/20 py-8 md:border-b-0 md:pr-8 ${i ? "md:border-l md:border-white/20 md:pl-8" : ""}`}>
                  <span className="font-mono text-[11px] text-[#f36c21] font-bold">{item.index} / 03</span>
                  <div className="mt-6 font-literary text-[32px] leading-none text-white">{item.title}</div>
                  <p className="mt-4 max-w-[340px] text-[13px] leading-[1.75] text-gray-300">{item.description}</p>
                  <span className="mt-auto pt-6 font-mono text-[10px] uppercase tracking-[.13em] text-[#f36c21]">{item.meta}</span>
                </article>
              ))}
            </div>
          </div>
        </section>

        {/* Assignment Requirements & Download */}
        <section id="project" className="scroll-mt-[76px] py-16 sm:py-22">
          <div className="mx-auto grid max-w-[1480px] gap-12 px-5 sm:px-8 lg:grid-cols-[.95fr_1.05fr] lg:gap-20 lg:px-14">
            <div>
              <div className="mb-3 font-mono text-[10px] uppercase tracking-[.16em] text-[#f36c21]">OFFICIAL DELIVERABLE / 03</div>
              <h2 className="font-literary text-[clamp(3rem,4.5vw,5.2rem)] leading-[.95] text-foreground">
                Rayvat Project<br/>Submission Spec<span className="text-[#f36c21]">.</span>
              </h2>
              <p className="mt-6 max-w-[490px] text-[14px] leading-[1.85] text-muted-foreground">
                Includes full modular Python codebase, trained model weights (`best_model.pt`), early stopping history plots (`training_loss.png`), sample suites (`generated_samples.txt`), and interactive notebooks.
              </p>
              
              <div className="mt-8 flex flex-wrap gap-4">
                <Button asChild className="h-11 rounded-sm bg-[#004b87] px-5 text-[12px] font-bold uppercase tracking-[.09em] text-white hover:bg-[#003866]">
                  <a href="/downloads/lstm_text_generator.py" download><Download className="mr-2 size-4"/> Download Python Code</a>
                </Button>
              </div>
            </div>

            <div className="border-t border-line">
              <div className="flex items-center justify-between border-b border-line py-4">
                <span className="font-mono text-[10px] uppercase tracking-[.14em] text-muted-foreground">MODEL SPECIFICATION BLUEPRINT</span>
                <span className="font-mono text-[10px] font-bold text-[#f36c21]">PYTORCH 2.12</span>
              </div>
              {[
                ["01", "Dataset Preprocessing", "Gutenberg metadata stripped · lowercasing · punctuation removal"],
                ["02", "Word Vocabulary", "12,850 words · reserved <PAD> (0) and <UNK> (1) tokens"],
                ["03", "Sequence Window", "30 words input context -> target 31st next word"],
                ["04", "Stacked LSTM Layer", "2 stacked LSTM layers · 256 hidden dimension · 0.2 dropout"],
                ["05", "Training & Validation", "90/10 sequential split · Adam (lr=0.001) · CrossEntropyLoss"],
                ["06", "Early Stopping & Perplexity", "Automatic checkpointing (best_model.pt) · Train & Val Perplexity tracking"]
              ].map(([number, name, detail]) => (
                <div key={number} className="grid grid-cols-[35px_1fr] items-start gap-x-3 gap-y-1 border-b border-line py-4 sm:grid-cols-[35px_160px_1fr] sm:items-center">
                  <span className="font-mono text-[11px] font-bold text-[#f36c21]">{number}</span>
                  <span className="text-[14px] font-semibold text-foreground">{name}</span>
                  <span className="col-start-2 text-[12px] text-muted-foreground sm:col-auto">{detail}</span>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-line bg-[#001d36] text-white">
        <div className="mx-auto flex max-w-[1480px] flex-wrap items-center justify-between gap-4 px-5 py-8 sm:px-8 lg:px-14">
          <div className="flex items-center gap-3">
            <span className="font-extrabold text-xl text-white"><span className="text-[#f36c21]">R</span>AYVAT</span>
            <span className="font-mono text-[10px] uppercase tracking-[.14em] text-gray-400">| OUTSOURCING AI LABS</span>
          </div>
          <span className="font-mono text-[10px] uppercase tracking-[.1em] text-gray-400">
            OFFICIAL ASSIGNMENT PORTAL • 2026
          </span>
          <a href="#top" className="flex items-center gap-1 text-[11px] text-gray-300 hover:text-white">
            Back to top <ChevronDown className="size-3 rotate-180"/>
          </a>
        </div>
      </footer>
    </div>
  );
}
