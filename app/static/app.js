let page = 1, query = "", category = "", currentAsin = null, starSel = 0;
let searchTimer = null;

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c =>
  ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const fmtPrice = (p) => (p && !isNaN(p)) ? `$${Number(p).toFixed(2)}` : "";
const show = (id) => $(id).classList.remove("hidden");
const hide = (id) => $(id).classList.add("hidden");

async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: { "Content-Type": "application/json" }, ...opts });
  if (r.status === 401 && path !== "/api/me") { show("authModal"); throw new Error("auth"); }
  return r.json();
}

async function boot() {
  try {
    const me = await api("/api/me");
    $("whoami").textContent = me.username;
    $("authBtn").textContent = "Sign out";
    loadRecs();
  } catch { $("whoami").textContent = ""; }
  loadCategories();
  loadProducts();
}

function authAction() {
  if ($("authBtn").textContent === "Sign out") {
    api("/api/logout", { method: "POST" }).then(() => location.reload());
  } else show("authModal");
}

async function doLogin() {
  const b = JSON.stringify({ username: $("username").value.trim(),
                             password: $("password").value });
  const r = await api("/api/login", { method: "POST", body: b });
  if (r.ok) location.reload(); else $("authMsg").textContent = r.detail || "login failed";
}
async function doSignup() {
  const b = JSON.stringify({ username: $("username").value.trim(),
                             password: $("password").value });
  const r = await api("/api/signup", { method: "POST", body: b });
  if (r.ok) location.reload(); else $("authMsg").textContent = r.detail || "signup failed";
}
async function demoLogin() {
  const r = await api("/api/demo", { method: "POST" });
  if (r.ok) location.reload();
}

function debouncedSearch() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    query = $("search").value.trim(); page = 1; loadProducts();
  }, 350);
}

async function loadCategories() {
  const { categories } = await api("/api/categories");
  $("catList").innerHTML =
    `<li onclick="setCat('')" class="active">All products</li>` +
    categories.map(c =>
      `<li onclick="setCat('${esc(c)}')">${esc(c)}</li>`).join("");
}
function setCat(c) {
  category = c; page = 1;
  document.querySelectorAll("#catList li").forEach(li =>
    li.classList.toggle("active",
      li.textContent === (c || "All products")));
  $("gridTitle").textContent = c || "Browse products";
  loadProducts();
}

async function loadProducts() {
  const d = await api(`/api/products?q=${encodeURIComponent(query)}` +
    `&category=${encodeURIComponent(category)}&page=${page}`);
  $("grid").innerHTML = d.products.map(card).join("");
  $("pageInfo").textContent = `page ${d.page} · ${d.total} items`;
}
const card = (p) => `
  <div class="card" onclick="openDetail('${esc(p.asin)}')">
    <img src="${esc(p.image)}" loading="lazy" onerror="this.style.visibility='hidden'">
    <div class="body">
      <div class="title">${esc(p.title)}</div>
      <div class="sub">${esc(p.brand)} · ${esc(p.category)}</div>
      <div class="price">${fmtPrice(p.price)}</div>
    </div>
  </div>`;

function nextPage() { page++; loadProducts(); }
function prevPage() { if (page > 1) { page--; loadProducts(); } }

async function loadRecs() {
  const d = await api("/api/recommendations?k=12");
  if (!d.recommendations?.length) return;
  show("recSection");
  $("recBadge").textContent = d.cold_start ? "cold-start mode" : "hybrid AI";
  $("recs").innerHTML = d.recommendations.map(r => `
    <div class="rec-card" onclick="openDetail('${esc(r.asin)}')">
      <img src="${esc(r.product?.image || "")}" loading="lazy"
           onerror="this.style.visibility='hidden'">
      <div class="body">
        <div class="title">${esc(r.product?.title || r.asin)}</div>
        ${r.reasons.map(x => `<span class="reason">${esc(x)}</span>`).join("")}
        <div class="score">match ${(r.score * 100).toFixed(0)}%</div>
      </div>
    </div>`).join("");
}

async function openDetail(asin) {
  currentAsin = asin; starSel = 0;
  const p = await api(`/api/products/${encodeURIComponent(asin)}`);
  $("dImg").src = p.image || "";
  $("dTitle").textContent = p.title;
  $("dBrand").textContent = p.brand || "";
  $("dCat").textContent = p.category || "";
  $("dPrice").textContent = fmtPrice(p.price);
  $("dDesc").textContent = p.description || "";
  $("dMsg").textContent = "";
  renderStars();
  $("similar").innerHTML = (p.similar_products || []).map(r => `
    <div class="rec-card" onclick="openDetail('${esc(r.asin)}')">
      <img src="${esc(r.image)}" loading="lazy" onerror="this.style.visibility='hidden'">
      <div class="body"><div class="title">${esc(r.title)}</div></div>
    </div>`).join("");
  show("detailModal");
}

function renderStars() {
  $("dStars").innerHTML = [1,2,3,4,5].map(i =>
    `<span class="${i <= starSel ? "on" : ""}" onclick="starSel=${i};renderStars()">★</span>`
  ).join("");
}

async function rate() {
  if (!starSel) { $("dMsg").textContent = "Pick a star rating first."; return; }
  await api("/api/interactions", { method: "POST", body: JSON.stringify(
    { asin: currentAsin, event: "rate", rating: starSel }) });
  $("dMsg").textContent = `Rated ${starSel}/5 - your recommendations updated.`;
  loadRecs();
}
async function purchase() {
  await api("/api/interactions", { method: "POST", body: JSON.stringify(
    { asin: currentAsin, event: "purchase" }) });
  $("dMsg").textContent = "Purchase recorded - recommendations updated.";
  loadRecs();
}

boot();
