document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("[data-consensus-form]");
  if (form) {
    form.querySelectorAll("[data-consensus-field]").forEach(section => {
      const inputs = [...section.querySelectorAll("[data-consensus-value]")];
      const reason = section.querySelector("[data-consensus-reason]");
      const apply = (choice, initial = false) => {
        const custom = choice?.value === "custom";
        const content = choice && !custom ? JSON.parse(choice.dataset.value) : null;
        inputs.forEach(input => {
          if (!initial && content !== null) {
            if (input.type === "checkbox") input.checked = content.includes(input.value);
            else input.value = content;
          }
          if (input.tagName === "TEXTAREA") input.readOnly = !custom;
          else input.disabled = !custom;
        });
        if (reason) {
          if (!initial && choice && !custom) reason.value = choice.dataset.reason || "";
          reason.readOnly = !custom;
        }
      };
      if (section.dataset.agreed === "yes") return;
      const selected = section.querySelector("[data-consensus-choice]:checked");
      apply(selected, true);
      section.querySelectorAll("[data-consensus-choice]").forEach(choice => {
        choice.addEventListener("change", () => {
          apply(choice);
          form.dispatchEvent(new Event("input", { bubbles: true }));
        });
      });
    });
  }
  const waiting = document.querySelector("[data-audit-wait-url]");
  if (waiting) {
    const check = async () => {
      if (document.hidden || document.querySelector("[data-documents-panel]:not([hidden])")) return;
      try {
        const response = await fetch(waiting.dataset.auditWaitUrl, { credentials: "same-origin" });
        if (!response.ok) return;
        const data = await response.json();
        if (data.next) window.location.replace(data.next);
      } catch { /* Leave the waiting page usable when offline. */ }
    };
    window.setInterval(check, 15000);
    document.addEventListener("visibilitychange", check);
  }
});
