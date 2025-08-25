---
layout: default
title: Papers
permalink: /papers/
---

# Papers

<div class="papers-grid">
  {% assign papers = site.papers | sort: "year" | reverse %}
  {% for p in papers %}
  <article class="paper-card">
    <h3 class="paper-title">
      <a href="{{ p.link }}" target="_blank" rel="noopener noreferrer">
        {{ p.title }}
      </a>
    </h3>
    <div class="paper-meta">
      <span class="year">{{ p.year }}</span><br>
    </div>
    <p class="paper-abstract">
      {{ p.content }}
    </p>
  </article>
  {% endfor %}
</div>
