document.addEventListener("DOMContentLoaded", () => {
    const root = document.documentElement;
    const themeToggle = document.querySelector("[data-theme-toggle]");
    const themeIcon = themeToggle?.querySelector(".theme-toggle-icon");
    const themeLabel = themeToggle?.querySelector(".theme-toggle-label");

    const storedTheme = (() => {
        try {
            return localStorage.getItem("aegis-credit-theme");
        } catch {
            return null;
        }
    })();

    const systemPrefersDark = window.matchMedia?.("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = storedTheme || (systemPrefersDark ? "dark" : "light");

    const updateThemeToggle = () => {
        const isDark = root.dataset.theme === "dark";
        if (themeToggle) {
            themeToggle.setAttribute("aria-pressed", String(isDark));
            themeToggle.setAttribute("aria-label", isDark ? "Enable light mode" : "Enable dark mode");
        }
        if (themeIcon) {
            themeIcon.classList.toggle("is-dark", isDark);
        }
        if (themeLabel) {
            themeLabel.textContent = isDark ? "Light" : "Dark";
        }
    };

    updateThemeToggle();
    themeToggle?.addEventListener("click", () => {
        const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
        root.dataset.theme = nextTheme;
        try {
            localStorage.setItem("aegis-credit-theme", nextTheme);
        } catch {
            // The visual setting still works when browser storage is unavailable.
        }
        updateThemeToggle();
    });

    for (const group of document.querySelectorAll(".field-group")) {
        const control = group.querySelector("input:not([type='hidden']), select, textarea");
        if (!control) {
            continue;
        }
        const descriptions = Array.from(group.querySelectorAll(".field-hint[id], .field-error[id]"));
        const describedBy = new Set((control.getAttribute("aria-describedby") || "").split(/\s+/).filter(Boolean));
        for (const description of descriptions) {
            describedBy.add(description.id);
        }
        if (describedBy.size) {
            control.setAttribute("aria-describedby", Array.from(describedBy).join(" "));
        }
        if (group.querySelector(".field-error")) {
            control.setAttribute("aria-invalid", "true");
        }
    }

    for (const form of document.querySelectorAll("[data-loading-form]")) {
        form.addEventListener("submit", () => {
            if (!form.checkValidity()) {
                return;
            }
            const button = form.querySelector('button[type="submit"]');
            if (button) {
                button.classList.add("is-loading");
                button.setAttribute("aria-busy", "true");
                button.setAttribute("aria-label", button.dataset.loadingLabel || "Processing request");
                button.disabled = true;
            }
        });
    }

    for (const slider of document.querySelectorAll("[data-threshold-slider]")) {
        const output = document.getElementById(slider.getAttribute("aria-controls"));
        if (!output) {
            continue;
        }
        const updateOutput = () => {
            output.value = Number(slider.value).toFixed(2);
            output.textContent = output.value;
        };
        slider.addEventListener("input", updateOutput);
        updateOutput();
    }

    for (const button of document.querySelectorAll("[data-print-button]")) {
        button.addEventListener("click", () => window.print());
    }
});
