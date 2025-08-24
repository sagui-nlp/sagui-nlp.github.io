---
layout: default
title: Members
---

# Members

<div class="members-grid">
  {% for m in site.members %}
  <figure>
    <img
      class="avatar"
      src="{{ m.avatar | relative_url }}"
      alt="{{ m.name }}"
      width="160" height="160"
      loading="lazy">
    <figcaption>
      <a class="member-name" href="{{ m.linkedin }}" target="_blank" rel="noopener noreferrer">
        {{ m.name }}
      </a>
    </figcaption>
  </figure>
  {% endfor %}
</div>

