ADMIN_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Game Admin</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: #111; color: #ddd; }
    #panel {
      position: sticky; top: 0; z-index: 10;
      padding: 10px 14px; background: #111; border-bottom: 1px solid #333;
    }
    .row { margin: 6px 0; font-size: 15px; }
    #turn { font-size: 18px; }
    .label { color: #777; font-size: 13px; }
    .player { color: #4fc; font-weight: bold; }
    .chip  { color: #ffd166; font-weight: bold; }
    .cell  { color: #fc4; }
    .fsm   { color: #4af; }
    .rule  { color: #f84; }
    #cols { margin-top: 10px; display: flex; gap: 14px; flex-wrap: wrap; align-items: flex-start; }
    .col { display: flex; flex-direction: column; gap: 6px; }
    .col .ch { font-size: 11px; color: #666; text-transform: uppercase; margin-bottom: 2px; }
    button {
      padding: 9px 14px; font-size: 14px; cursor: pointer; text-align: left;
      background: #222; color: #ddd; border: 1px solid #444; border-radius: 6px;
    }
    button:active { background: #444; }
    button.danger { border-color: #a33; color: #f88; }
    button.go { border-color: #2a6; color: #6fc; }
    select#pathSel { padding: 9px 10px; font-size: 13px; background: #222; color: #ddd;
      border: 1px solid #444; border-radius: 6px; font-family: monospace; cursor: pointer; }
    #panelToggle {
      position: fixed; top: 8px; right: 8px; z-index: 400;
      padding: 6px 11px; font-size: 13px; font-family: monospace; cursor: pointer;
      background: rgba(20,20,20,0.85); color: #ddd; border: 1px solid #555; border-radius: 6px;
    }
    /* dynamic admin panel (list/delete/images/add/remove player) */
    #dynPanel { margin-top: 10px; padding: 10px; background: #1a1a1a; border: 1px solid #444; border-radius: 6px; display: none; }
    #dynPanel .ttl { font-size: 14px; color: #4fc; margin-bottom: 8px; }
    #dynPanel input { background: #222; color: #ddd; border: 1px solid #444; border-radius: 4px; padding: 8px 10px; font-size: 14px; font-family: monospace; width: 100%; margin-bottom: 8px; }
    .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(90px, 1fr)); gap: 8px; }
    .card { border: 2px solid #333; border-radius: 8px; padding: 6px; background: #161616; text-align: center; position: relative; }
    .card.sel { border-color: #2a6; background: #163; cursor: pointer; }
    .card.pick { cursor: pointer; }
    .card img { width: 100%; height: 64px; object-fit: contain; background: #000; border-radius: 4px; }
    .card .nm { font-size: 12px; color: #ffd166; margin-top: 4px; word-break: break-word; }
    .card .del { position: absolute; top: 2px; right: 2px; background: #a33; color: #fff; border: none; border-radius: 4px; padding: 2px 7px; cursor: pointer; font-size: 13px; }
    .prow { display: flex; align-items: center; gap: 8px; padding: 4px 0; border-bottom: 1px solid #2a2a2a; }
    .prow img { width: 36px; height: 36px; object-fit: contain; background: #000; border-radius: 4px; }
    .prow .rm { margin-left: auto; background: #a33; color: #fff; border: none; border-radius: 4px; padding: 4px 10px; cursor: pointer; }
    #joinurl { margin-top: 8px; font-size: 13px; color: #555; }
    #joinurl a { color: #4af; text-decoration: none; }
    #connect { margin-top: 10px; display: flex; align-items: center; gap: 12px; }
    #qr { width: 132px; height: 132px; background: #fff; padding: 6px; border-radius: 6px; display: none; }
    #connectText { font-size: 13px; color: #888; }
    #connectText a { color: #4af; text-decoration: none; word-break: break-all; }
    /* login overlay */
    #login {
      position: fixed; inset: 0; z-index: 500; display: flex;
      flex-direction: column; align-items: center; justify-content: center;
      background: #111; padding: 24px; text-align: center;
    }
    #login h2 { color: #4fc; margin-bottom: 18px; }
    #login input {
      width: 240px; padding: 12px; font-size: 16px; font-family: monospace;
      background: #222; color: #ddd; border: 1px solid #444; border-radius: 6px; margin-bottom: 12px;
    }
    #login button {
      width: 240px; padding: 12px; font-size: 15px; font-family: monospace;
      background: #2a6; color: #fff; border: none; border-radius: 6px; cursor: pointer;
    }
    #loginErr { color: #f64; font-size: 14px; margin-top: 10px; min-height: 18px; }
    #loginJoinUrl a { color: #4af; text-decoration: none; word-break: break-all; }
    /* registration panel */
    #regPanel { margin-top: 10px; padding: 10px; background: #1a1a1a; border: 1px solid #444; border-radius: 6px; display: none; }
    #regPanel input { background: #222; color: #ddd; border: 1px solid #444; border-radius: 4px; padding: 6px 10px; font-size: 14px; font-family: monospace; width: 180px; }
    #regPanel button { padding: 6px 14px; font-size: 14px; }
    #regHint { margin-top: 6px; font-size: 12px; color: #888; }
    #regProgress { margin-top: 6px; font-size: 15px; font-weight: bold; color: #4af; }
    /* notification banner */
    #notification {
      position: fixed; top: 0; left: 0; right: 0; z-index: 200;
      padding: 14px 20px; text-align: center;
      background: #e85; color: #fff; font-size: 16px; font-weight: bold;
      display: none; transition: opacity 0.4s;
    }
    /* feed wrapper for canvas overlay */
    #feedWrap { position: relative; }
    #feed { width: 100%; display: block; }
    #roiCanvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: none; cursor: crosshair; }
  </style>
</head>
<body>
  <div id="login">
    <h2 style="margin-bottom:12px">Подключение игроков</h2>
    <img id="loginQr" alt="QR для подключения" style="width:180px;height:180px;background:#fff;padding:8px;border-radius:8px;display:none">
    <div id="loginJoinUrl" style="margin-top:10px;font-size:14px"></div>
    <div style="margin-top:4px;color:#555;font-size:13px">Наведите камеру телефона на QR</div>
    <hr style="width:240px;border:none;border-top:1px solid #333;margin:22px 0">
    <h2>Кабинет ведущего</h2>
    <input id="loginPw" type="password" placeholder="Пароль" autocomplete="current-password">
    <button onclick="doLogin()">Войти</button>
    <div id="loginErr"></div>
  </div>
  <div id="notification"></div>
  <button id="panelToggle" onclick="togglePanel()">Скрыть панель</button>
  <div id="panel">
    <div class="row" id="turn"></div>
    <div class="row" id="players"></div>
    <div id="cols">
      <div class="col">
        <div class="ch">Фишки</div>
        <button data-act="register">&#9998; Регистрация</button>
        <button data-act="list">&#9776; Список фишек</button>
        <button data-act="delete">&#10005; Удалить</button>
      </div>
      <div class="col">
        <div class="ch">Игроки / Поле</div>
        <button data-act="addplayer">+ Игрок</button>
        <button data-act="removeplayer">&minus; Игрок</button>
        <button data-act="grid">&#9638; Сетка вкл/выкл</button>
        <button data-act="calibrate">&#9712; Калибровка</button>
        <select id="pathSel" title="Порядок обхода клеток">
          <option value="linear">Маршрут: линейный</option>
          <option value="snake">Маршрут: змейка</option>
          <option value="spiral">Маршрут: спираль</option>
        </select>
      </div>
      <div class="col">
        <div class="ch">Игра</div>
        <button data-act="start" id="startBtn" class="go">&#9654;&#9654; Старт игры</button>
        <button data-act="next" class="go">&#9654; Следующий ход</button>
        <button data-act="endgame">&#9632; Стоп партии</button>
      </div>
      <div class="col">
        <div class="ch">Система</div>
        <button data-act="overlay">&#9783; Панель cv2 вкл/выкл</button>
        <button data-act="stop" class="danger">&#9211; Выключить систему</button>
      </div>
    </div>

    <!-- регистрация (drag по стриму) -->
    <div id="regPanel" style="margin-top:10px;padding:10px;background:#1a1a1a;border:1px solid #444;border-radius:6px;display:none">
      <input id="regName" placeholder="Название фишки" autocomplete="off" style="background:#222;color:#ddd;border:1px solid #444;border-radius:4px;padding:6px 10px;font-size:14px;font-family:monospace;width:180px">
      <button onclick="startReg()">Начать</button>
      <button onclick="cancelReg()" style="border-color:#f64;color:#f64">Отмена</button>
      <div id="regHint" style="margin-top:6px;font-size:12px;color:#888">Введите название и нажмите «Начать». Затем обведите фишку рамкой прямо на видео — нужно сделать несколько снимков с разных положений.</div>
      <div id="regProgress" style="margin-top:6px;font-size:15px;font-weight:bold;color:#4af"></div>
    </div>

    <!-- динамическая панель (список/удаление/картинки/игроки) -->
    <div id="dynPanel"></div>

    <div id="connect">
      <img id="qr" alt="QR для подключения">
      <div id="connectText">
        <div>Подключение игроков:</div>
        <div id="joinurl"></div>
        <div style="margin-top:4px;color:#555">Наведите камеру телефона на QR</div>
      </div>
    </div>
  </div>
  <div id="feedWrap">
    <img id="feed" src="/stream" alt="Game feed">
    <canvas id="roiCanvas"></canvas>
  </div>
  <script>
    // --- вход в кабинет ведущего ---
    function showPanel() {
      document.getElementById('login').style.display = 'none';
      initConnect();
    }
    function doLogin() {
      const pw = document.getElementById('loginPw').value;
      fetch('/api/admin/login', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({password: pw})
      }).then(r => r.json()).then(d => {
        if (d.ok) { sessionStorage.setItem('adminOk', '1'); showPanel(); }
        else { document.getElementById('loginErr').textContent = 'Неверный пароль'; }
      }).catch(() => { document.getElementById('loginErr').textContent = 'Ошибка сети'; });
    }
    document.getElementById('loginPw').addEventListener('keydown', e => { if (e.key === 'Enter') doLogin(); });
    if (sessionStorage.getItem('adminOk') === '1') showPanel();

    // --- ссылка и QR для подключения (берём реальный LAN-IP с сервера) ---
    // qrId/urlId позволяют показать QR и на странице входа, и в панели после входа
    function loadConnect(qrId, urlId) {
      const urlEl = document.getElementById(urlId);
      const qr = document.getElementById(qrId);
      fetch('/api/netinfo').then(r => r.json()).then(d => {
        const host = d.ip || window.location.hostname;
        const joinUrl = 'http://' + host + ':' + (d.port || 5000) + '/join';
        if (urlEl) urlEl.innerHTML = '<a href="' + joinUrl + '" target="_blank">' + joinUrl + '</a>';
        if (qr) {
          qr.src = '/api/qr';
          qr.onload = () => { qr.style.display = 'block'; };
          qr.onerror = () => { qr.style.display = 'none'; };
        }
      }).catch(() => {
        const joinUrl = window.location.protocol + '//' + window.location.hostname +
          (window.location.port ? ':' + window.location.port : '') + '/join';
        if (urlEl) urlEl.innerHTML = '<a href="' + joinUrl + '" target="_blank">' + joinUrl + '</a>';
      });
    }
    function initConnect() { loadConnect('qr', 'joinurl'); }
    // QR на странице входа доступен сразу, до авторизации
    loadConnect('loginQr', 'loginJoinUrl');

    // свернуть/развернуть всю веб-панель управления (чтобы видеть поле целиком)
    function togglePanel() {
      const p = document.getElementById('panel');
      const hidden = p.style.display === 'none';
      p.style.display = hidden ? '' : 'none';
      document.getElementById('panelToggle').textContent = hidden ? 'Скрыть панель' : 'Показать панель';
    }

    // --- панель управления (4 столбца кнопок) ---
    const dyn = document.getElementById('dynPanel');
    const regPanel = document.getElementById('regPanel');
    let dynMode = null;
    let addPick = null;

    document.getElementById('cols').addEventListener('click', e => {
      const btn = e.target.closest('button[data-act]');
      if (btn) handleAct(btn.dataset.act);
    });

    // смена порядка обхода клеток поля
    document.getElementById('pathSel').addEventListener('change', e => {
      fetch('/api/game/path', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({type: e.target.value})
      }).then(r => r.json()).then(d => {
        if (d.error) showNotice(d.error);
      }).catch(() => {});
    });

    function closeDyn() { dyn.style.display = 'none'; dyn.innerHTML = ''; dynMode = null; }

    function handleAct(act) {
      // повторное нажатие Регистрации — скрыть панель (toggle)
      if (act === 'register') {
        if (regPanel.style.display === 'block') { cancelReg(); regPanel.style.display = 'none'; }
        else { closeDyn(); cancelCalib(); regPanel.style.display = 'block'; }
        return;
      }
      if (regActive) cancelReg();
      regPanel.style.display = 'none';
      // повторное нажатие той же панели — закрыть (toggle)
      if (act === dynMode || (act === 'list' && dynMode === 'gallery')) { cancelCalib(); closeDyn(); return; }
      if (act !== 'calibrate') cancelCalib();
      if (act !== 'list' && act !== 'delete' && act !== 'addplayer' &&
          act !== 'removeplayer' && act !== 'calibrate') closeDyn();
      if (act === 'next')      { cmd('next'); }
      else if (act === 'start'){ cmd('start'); }
      else if (act === 'grid') { cmd('grid'); }
      else if (act === 'overlay') { cmd('overlay'); }
      else if (act === 'endgame') { if (confirm('Остановить текущую партию?')) cmd('endgame'); }
      else if (act === 'stop') { if (confirm('Выключить систему и сервер? Веб перестанет работать.')) cmd('stop'); }
      else if (act === 'calibrate') { openCalibrate(); }
      else if (act === 'list') { openGallery(false); }
      else if (act === 'delete') { openGallery(true); }
      else if (act === 'addplayer') { openAddPlayer(); }
      else if (act === 'removeplayer') { openRemovePlayer(); }
    }

    function openGallery(deletable) {
      dynMode = deletable ? 'delete' : 'gallery';
      dyn.style.display = 'block';
      refreshGallery(deletable);
    }
    function refreshGallery(deletable) {
      fetch('/api/chips/all').then(r => r.json()).then(d => {
        const chips = d.chips || [];
        const ttl = deletable ? 'Удаление фишек' : 'Зарегистрированные фишки';
        if (!chips.length) { dyn.innerHTML = '<div class="ttl">' + ttl + '</div><div class="label">нет фишек</div>'; return; }
        dyn.innerHTML = '<div class="ttl">' + ttl + ' (' + chips.length + ')</div><div class="cards">' +
          chips.map(c => '<div class="card">' +
            (deletable ? '<button class="del" data-del="' + c.id + '">&#10005;</button>' : '') +
            '<img src="/api/chips/' + c.id + '/image" onerror="this.style.visibility=\\'hidden\\'">' +
            '<div class="nm">' + c.name + '</div></div>').join('') + '</div>';
      }).catch(() => {});
    }

    function openAddPlayer() {
      dynMode = 'addplayer'; addPick = null;
      dyn.style.display = 'block';
      dyn.innerHTML = '<div class="ttl">Добавить игрока</div>' +
        '<input id="apName" placeholder="Имя игрока" autocomplete="off">' +
        '<div class="label" style="margin-bottom:6px">Выберите фишку:</div>' +
        '<div class="cards" id="apCards"></div>' +
        '<button class="go" id="apBtn" style="margin-top:8px">Добавить</button>' +
        '<div id="apMsg" class="label" style="margin-top:6px"></div>';
      refreshAddCards();
      document.getElementById('apBtn').onclick = submitAddPlayer;
    }
    function refreshAddCards() {
      fetch('/api/chips').then(r => r.json()).then(d => {
        const chips = d.chips || [];
        const box = document.getElementById('apCards');
        if (!box) return;
        if (!chips.length) { box.innerHTML = '<div class="label">нет свободных фишек</div>'; return; }
        if (addPick && !chips.some(c => c.id === addPick)) addPick = null;
        box.innerHTML = chips.map(c => '<div class="card pick' + (c.id === addPick ? ' sel' : '') + '" data-pick="' + c.id + '">' +
          '<img src="/api/chips/' + c.id + '/image" onerror="this.style.visibility=\\'hidden\\'">' +
          '<div class="nm">' + c.name + '</div></div>').join('');
      }).catch(() => {});
    }
    function submitAddPlayer() {
      const name = document.getElementById('apName').value.trim();
      const msg = document.getElementById('apMsg');
      if (!name) { msg.textContent = 'Введите имя'; return; }
      if (!addPick) { msg.textContent = 'Выберите фишку'; return; }
      fetch('/api/game/join', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name, chip_id: addPick})})
        .then(r => r.json()).then(res => {
          if (res.status === 'ok') { document.getElementById('apName').value=''; addPick=null; msg.textContent='Добавлен: '+name; refreshAddCards(); }
          else { msg.textContent = res.error || 'Ошибка'; }
        }).catch(() => { msg.textContent = 'Ошибка сети'; });
    }

    function openRemovePlayer() {
      dynMode = 'removeplayer';
      dyn.style.display = 'block';
      refreshRemove();
    }
    function refreshRemove() {
      fetch('/api/game/state').then(r => r.json()).then(d => {
        const q = (d.state || {}).queue || [];
        if (!q.length) { dyn.innerHTML = '<div class="ttl">Удалить игрока</div><div class="label">очередь пуста</div>'; return; }
        dyn.innerHTML = '<div class="ttl">Удалить игрока</div>' + q.map(p =>
          '<div class="prow"><img src="/api/chips/' + p.chip_id + '/image" onerror="this.style.visibility=\\'hidden\\'">' +
          '<span class="player">' + p.player + '</span> <span class="chip">' + (p.chip_name||'') + '</span>' +
          '<button class="rm" data-rmplayer="' + encodeURIComponent(p.player) + '">Убрать</button></div>').join('');
      }).catch(() => {});
    }

    // делегирование кликов внутри dynPanel
    dyn.addEventListener('click', e => {
      const del = e.target.closest('[data-del]');
      if (del) { fetch('/api/chips/delete', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({id: del.dataset.del})}).then(() => setTimeout(() => refreshGallery(true), 200)); return; }
      const rm = e.target.closest('[data-rmplayer]');
      if (rm) { fetch('/api/players/remove', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({value: decodeURIComponent(rm.dataset.rmplayer)})}).then(() => setTimeout(refreshRemove, 200)); return; }
      const pick = e.target.closest('[data-pick]');
      if (pick) { addPick = pick.dataset.pick; document.querySelectorAll('#apCards .card').forEach(el => el.classList.toggle('sel', el.dataset.pick === addPick)); return; }
    });

    // автообновление открытой панели
    setInterval(() => {
      if (dynMode === 'gallery') refreshGallery(false);
      else if (dynMode === 'delete') refreshGallery(true);
      else if (dynMode === 'addplayer') refreshAddCards();
      else if (dynMode === 'removeplayer') refreshRemove();
    }, 1500);

    // --- game state ---
    function update() {
      fetch('/api/game/state').then(r => r.json()).then(data => {
        const s = data.state || {};
        // отразить реальный маршрут поля в выпадающем списке (не мешая выбору пользователя)
        const psel = document.getElementById('pathSel');
        if (s.path_type && psel && document.activeElement !== psel && psel.value !== s.path_type) {
          psel.value = s.path_type;
        }
        if (!s.active) {
          const winner = s.winner
            ? 'GAME OVER: <span class="player">' + s.winner + '</span> wins'
            : 'Game not started';
          document.getElementById('turn').innerHTML = '<span class="fsm">' + winner + '</span>';
          document.getElementById('players').innerHTML = '';
          document.getElementById('startBtn').style.display = '';
          return;
        }
        document.getElementById('startBtn').style.display = 'none';
        const pl = (s.players || []).map(p =>
          '<span class="player">' + p.name + '</span>&nbsp;<span class="chip">' + (p.chip_name||'') +
          '</span>&nbsp;<span class="label">cell</span>&nbsp;<span class="cell">' + p.cell + '</span>'
        ).join('&emsp;');
        document.getElementById('players').innerHTML = pl;
        const cur = s.current_turn;
        let t = cur
          ? 'Turn: <span class="player">' + cur.player + '</span>&nbsp;<span class="chip">[' + cur.chip_name + ']</span>'
          : '<span class="fsm">Game ready: press Next Turn</span>';
        if (s.turn_state === 'WAITING_MOVE' && s.expected_cell) {
          t += '&nbsp;<span class="fsm">&#127922;' + s.dice_roll + '</span>&nbsp;&rarr;&nbsp;cell&nbsp;<span class="cell">' + s.expected_cell + '</span>';
        } else if (s.turn_state === 'AWAITING_RULE' && s.rule_target) {
          t += '&nbsp;<span class="rule">rule&nbsp;&rarr;&nbsp;cell&nbsp;' + s.rule_target + '</span>';
        } else if (s.turn_state === 'TURN_DONE' && cur) {
          t += '&nbsp;<span class="label">(пропуск / ход сделан)</span>';
        }
        document.getElementById('turn').innerHTML = t;
      }).catch(() => {});
    }
    function cmd(name) {
      fetch('/api/game/' + name, { method: 'POST' }).catch(() => {});
    }
    setInterval(update, 1000);
    update();

    // --- уведомления (NOTICE: напр. «нужно 2+ игроков») ---
    let noticeBaseline = -1;
    let noticeTimeout = null;
    function showNotice(text) {
      const el = document.getElementById('notification');
      el.textContent = text;
      el.style.display = 'block';
      el.style.opacity = '1';
      if (noticeTimeout) clearTimeout(noticeTimeout);
      noticeTimeout = setTimeout(() => {
        el.style.opacity = '0';
        setTimeout(() => { el.style.display = 'none'; }, 400);
      }, 5000);
    }
    function pollNotices() {
      fetch('/api/game/events').then(r => r.json()).then(d => {
        const evs = d.events || [];
        if (noticeBaseline < 0) { noticeBaseline = evs.length; return; }
        if (evs.length > noticeBaseline) {
          evs.slice(noticeBaseline).forEach(ev => {
            if (ev.type === 'NOTICE') showNotice((ev.data && ev.data.message) || 'Внимание');
          });
          noticeBaseline = evs.length;
        }
      }).catch(() => {});
    }
    setInterval(pollNotices, 1500);

    // --- registration via stream ---
    let regActive = false;
    let regTotal = 5;
    const canvas = document.getElementById('roiCanvas');
    const ctx = canvas.getContext('2d');
    let drag = null;


    function startReg() {
      const name = document.getElementById('regName').value.trim();
      if (!name) { document.getElementById('regProgress').textContent = 'Enter a name'; return; }
      fetch('/api/register/start', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name})
      }).then(r => r.json()).then(() => {
        regActive = true;
        activateCanvas();
        pollRegStatus();
      });
    }

    function cancelReg() {
      regActive = false;
      fetch('/api/register/cancel', {method: 'POST'}).catch(() => {});
      deactivateCanvas();
      document.getElementById('regProgress').textContent = '';
    }

    function activateCanvas() {
      const img = document.getElementById('feed');
      canvas.width = img.clientWidth;
      canvas.height = img.clientHeight;
      canvas.style.display = 'block';
    }

    function deactivateCanvas() {
      canvas.style.display = 'none';
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drag = null;
    }

    function pollRegStatus() {
      if (!regActive) return;
      fetch('/api/register/status').then(r => r.json()).then(d => {
        const reg = d.reg || {};
        if (reg.done) {
          document.getElementById('regProgress').textContent = '✓ Фишка «' + (reg.name||'') + '» зарегистрирована!';
          regActive = false;
          deactivateCanvas();
          setTimeout(() => { document.getElementById('regProgress').textContent = ''; }, 4000);
          return;
        }
        if (reg.active) {
          document.getElementById('regProgress').textContent =
            'Снимок ' + (reg.shot_num + 1) + ' из ' + reg.total + ' — обведите фишку рамкой на видео';
          regTotal = reg.total;
        } else {
          // команда start ещё не обработана сервером — ждём
          document.getElementById('regProgress').textContent = 'Запуск регистрации...';
        }
        setTimeout(pollRegStatus, 400);   // опрашиваем, пока активна регистрация на клиенте
      }).catch(() => { if (regActive) setTimeout(pollRegStatus, 1000); });
    }

    // canvas drag handlers
    canvas.addEventListener('mousedown', e => {
      const r = canvas.getBoundingClientRect();
      if (calibActive) { addCalibPoint(e.clientX - r.left, e.clientY - r.top); return; }
      if (!regActive) return;
      drag = {x: e.clientX - r.left, y: e.clientY - r.top};
    });
    canvas.addEventListener('mousemove', e => {
      if (!drag || !regActive) return;
      const r = canvas.getBoundingClientRect();
      const cx = e.clientX - r.left, cy = e.clientY - r.top;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = '#0ff'; ctx.lineWidth = 2;
      ctx.strokeRect(drag.x, drag.y, cx - drag.x, cy - drag.y);
    });
    canvas.addEventListener('mouseup', e => {
      if (!drag || !regActive) return;
      const r = canvas.getBoundingClientRect();
      const cx = e.clientX - r.left, cy = e.clientY - r.top;
      const x1 = Math.min(drag.x, cx) / canvas.width;
      const y1 = Math.min(drag.y, cy) / canvas.height;
      const x2 = Math.max(drag.x, cx) / canvas.width;
      const y2 = Math.max(drag.y, cy) / canvas.height;
      drag = null;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if ((x2 - x1) < 0.01 || (y2 - y1) < 0.01) return;
      fetch('/api/register/shot', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({x1, y1, x2, y2})
      }).catch(() => {});
    });
    // touch support
    function touchToMouse(ev, type) {
      if (!regActive && !calibActive) return;
      ev.preventDefault();
      const t = ev.touches[0] || ev.changedTouches[0];
      canvas.dispatchEvent(new MouseEvent(type, {clientX: t.clientX, clientY: t.clientY}));
    }
    canvas.addEventListener('touchstart', e => touchToMouse(e, 'mousedown'), {passive: false});
    canvas.addEventListener('touchmove',  e => touchToMouse(e, 'mousemove'), {passive: false});
    canvas.addEventListener('touchend',   e => touchToMouse(e, 'mouseup'),   {passive: false});

    // --- калибровка поля через стрим (4 угла) ---
    let calibActive = false;
    let calibPoints = [];   // нормированные [x,y]
    const CALIB_LABELS = ['верх-лево (TL)', 'верх-право (TR)', 'низ-право (BR)', 'низ-лево (BL)'];

    function openCalibrate() {
      calibActive = true;
      calibPoints = [];
      activateCanvas();
      dyn.style.display = 'block';
      dynMode = null;   // не автообновляется
      renderCalib();
    }
    function cancelCalib() {
      if (!calibActive) return;
      calibActive = false;
      deactivateCanvas();
    }
    function renderCalib() {
      const n = calibPoints.length;
      let html = '<div class="ttl">Калибровка поля</div>';
      if (n < 4) {
        html += '<div class="label">Кликните угол ' + (n + 1) + ' из 4: <b>' + CALIB_LABELS[n] + '</b></div>';
      } else {
        html += '<div class="label">4 угла отмечены. Размер сетки (клетки):</div>' +
          '<div style="display:flex;gap:6px;margin:6px 0">' +
          '<input id="calCols" type="number" min="1" value="8" style="width:70px">' +
          '<span style="align-self:center">×</span>' +
          '<input id="calRows" type="number" min="1" value="5" style="width:70px"></div>' +
          '<button class="go" id="calApply">Применить</button>';
      }
      html += ' <button id="calCancel" style="border-color:#f64;color:#f64">Отмена</button>';
      dyn.innerHTML = html;
      const ap = document.getElementById('calApply');
      if (ap) ap.onclick = applyCalib;
      document.getElementById('calCancel').onclick = () => { cancelCalib(); closeDyn(); };
    }
    function addCalibPoint(px, py) {
      if (calibPoints.length >= 4) return;
      calibPoints.push([px / canvas.width, py / canvas.height]);
      drawCalib();
      renderCalib();
    }
    function drawCalib() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#fa0'; ctx.strokeStyle = '#fa0'; ctx.lineWidth = 2;
      ctx.font = '14px monospace';
      calibPoints.forEach((p, i) => {
        const x = p[0] * canvas.width, y = p[1] * canvas.height;
        ctx.beginPath(); ctx.arc(x, y, 6, 0, 7); ctx.fill();
        ctx.fillText(['TL','TR','BR','BL'][i], x + 9, y - 6);
      });
      if (calibPoints.length === 4) {
        ctx.beginPath();
        calibPoints.forEach((p, i) => {
          const x = p[0]*canvas.width, y = p[1]*canvas.height;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.closePath(); ctx.stroke();
      }
    }
    function applyCalib() {
      const cols = parseInt(document.getElementById('calCols').value, 10);
      const rows = parseInt(document.getElementById('calRows').value, 10);
      if (!cols || !rows || cols < 1 || rows < 1) { showNotice('Неверный размер сетки'); return; }
      fetch('/api/calibrate', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({corners: calibPoints, cols, rows})
      }).then(r => r.json()).then(() => {
        showNotice('Поле откалибровано: ' + cols + '×' + rows);
        cancelCalib();
        closeDyn();
      }).catch(() => showNotice('Ошибка калибровки'));
    }
  </script>
</body>
</html>
""".encode("utf-8")


JOIN_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Join Game</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: #111; color: #ddd; padding: 24px 20px; }
    h2 { color: #4fc; margin-bottom: 20px; font-size: 20px; }
    #lblName, #lblChip { display: block; margin-bottom: 6px; font-size: 13px; color: #777; }
    #lblChip { margin-top: 8px; }
    input {
      width: 100%; padding: 12px; font-size: 16px;
      background: #222; color: #ddd; border: 1px solid #444; border-radius: 6px;
      margin-bottom: 8px; font-family: monospace;
    }
    #chipGrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(90px, 1fr)); gap: 8px; margin-bottom: 16px; }
    .chipCard {
      border: 2px solid #333; border-radius: 8px; padding: 6px; cursor: pointer;
      background: #1a1a1a; text-align: center; transition: border-color 0.15s;
    }
    .chipCard.sel { border-color: #2a6; background: #163; }
    .chipCard img { width: 100%; height: 70px; object-fit: contain; border-radius: 4px; background: #000; }
    .chipCard .nm { font-size: 12px; margin-top: 4px; color: #ffd166; word-break: break-word; }
    .empty { color: #777; font-size: 14px; grid-column: 1/-1; }
    #joinBtn {
      width: 100%; padding: 14px; font-size: 16px; cursor: pointer;
      background: #2a6; color: #fff; border: none; border-radius: 6px; font-family: monospace;
    }
    #joinBtn:active { background: #195; }
    #joinBtn:disabled { background: #333; color: #666; cursor: default; }
    #error { color: #f64; margin-bottom: 14px; font-size: 14px; display: none; }
    #status { color: #777; font-size: 14px; margin-top: 10px; text-align: center; }
  </style>
</head>
<body>
  <h2 id="title">Join Game</h2>
  <div id="error"></div>
  <span id="lblName">Your name</span>
  <input id="name" type="text" placeholder="Enter your name" autocomplete="off" autocapitalize="words">
  <span id="lblChip">Your chip</span>
  <div id="chipGrid"><div class="empty">...</div></div>
  <button id="joinBtn" onclick="join()">Join &rarr;</button>
  <div id="status"></div>
  <script>
    let M = {};
    let selectedChip = null;

    fetch('/api/messages').then(r => r.json()).then(d => {
      M = d.messages || {};
      document.title = M.join_title || document.title;
      document.getElementById('title').textContent = M.join_title || 'Join Game';
      document.getElementById('lblName').textContent = M.join_name_label || 'Your name';
      document.getElementById('name').placeholder = M.join_name_placeholder || '';
      document.getElementById('lblChip').textContent = M.join_chip_label || 'Your chip';
      document.getElementById('joinBtn').textContent = M.join_btn || 'Join';
    }).catch(() => {});

    fetch('/api/game/state').then(r => r.json()).then(d => {
      if ((d.state || {}).active) {
        const err = document.getElementById('error');
        err.textContent = M.join_game_active || 'Game in progress';
        err.style.display = 'block';
        document.getElementById('joinBtn').disabled = true;
      }
    }).catch(() => {});

    function loadChips() {
      fetch('/api/chips').then(r => r.json()).then(data => {
        const grid = document.getElementById('chipGrid');
        const chips = data.chips || [];
        if (!chips.length) {
          grid.innerHTML = '<div class="empty">' + (M.join_no_chips || 'No chips available') + '</div>';
          selectedChip = null;
          return;
        }
        // если выбранная фишка занята кем-то — снять выбор
        if (selectedChip && !chips.some(c => c.id === selectedChip)) selectedChip = null;
        grid.innerHTML = chips.map(c =>
          '<div class="chipCard' + (c.id === selectedChip ? ' sel' : '') + '" data-id="' + c.id + '" onclick="pick(\\'' + c.id + '\\')">' +
            '<img src="/api/chips/' + c.id + '/image" alt="" onerror="this.style.visibility=\\'hidden\\'">' +
            '<div class="nm">' + c.name + '</div>' +
          '</div>'
        ).join('');
      }).catch(() => {});
    }
    function pick(id) {
      selectedChip = id;
      document.querySelectorAll('.chipCard').forEach(el => {
        el.classList.toggle('sel', el.dataset.id === id);
      });
    }
    loadChips();
    setInterval(loadChips, 1000);

    function join() {
      const name = document.getElementById('name').value.trim();
      const err = document.getElementById('error');
      const status = document.getElementById('status');
      if (!name) { err.textContent = M.join_error_name || 'Enter your name'; err.style.display = 'block'; return; }
      if (!selectedChip) { err.textContent = M.join_error_chip || 'Select a chip'; err.style.display = 'block'; return; }
      err.style.display = 'none';
      status.textContent = M.join_joining || 'Joining...';
      fetch('/api/game/join', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, chip_id: selectedChip })
      }).then(r => r.json()).then(data => {
        if (data.status === 'ok') {
          localStorage.setItem('playerName', name);
          window.location.href = '/game';
        } else {
          err.textContent = data.error || (M.join_error_network || 'Error');
          err.style.display = 'block';
          status.textContent = '';
        }
      }).catch(() => {
        err.textContent = M.join_error_network || 'Network error';
        err.style.display = 'block';
        status.textContent = '';
      });
    }

    document.getElementById('name').addEventListener('keydown', e => {
      if (e.key === 'Enter') join();
    });
  </script>
</body>
</html>
""".encode("utf-8")


GAME_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Game</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: #111; color: #ddd; }
    #panel {
      position: sticky; top: 0; z-index: 10;
      padding: 10px 14px; background: #111; border-bottom: 1px solid #333;
    }
    #feed { width: 100%; display: block; }
    .row { margin: 4px 0; font-size: 15px; }
    .label { color: #777; font-size: 13px; }
    .player { color: #4fc; font-weight: bold; }
    .chip { color: #ffd166; }
    .cell { color: #fc4; }
    .fsm { color: #4af; }
    .rule { color: #f84; }
    .you { color: #ff9; font-size: 12px; }
    #turn { font-size: 18px; margin-bottom: 4px; }
    #players { display: flex; flex-direction: column; gap: 3px; margin: 6px 0; }
    .player-row { font-size: 14px; padding: 2px 0; }
    #who { font-size: 13px; color: #666; margin: 4px 0 10px; line-height: 1.5; }
    #myturn { margin-top: 4px; }
    #nextBtn {
      padding: 14px 28px; font-size: 16px;
      border: 1px solid transparent; border-radius: 8px;
      font-family: monospace; width: 100%; cursor: default;
      transition: background 0.2s;
    }
    #nextBtn.btn-ready { background: #2a6; color: #fff; cursor: pointer; border-color: #2a6; }
    #nextBtn.btn-ready:active { background: #195; }
    #nextBtn.btn-moving { background: #332200; color: #fa0; border-color: #fa0; }
    #nextBtn.btn-waiting { background: #1a1a1a; color: #444; }
    /* notification banner */
    #notification {
      position: fixed; top: 0; left: 0; right: 0; z-index: 200;
      padding: 14px 20px; text-align: center;
      background: #e85; color: #fff; font-size: 16px; font-weight: bold;
      display: none; transition: opacity 0.4s;
    }
    /* finish overlay */
    #finish {
      position: fixed; inset: 0; z-index: 300; display: none;
      background: #0c0c0c; flex-direction: column; align-items: center; justify-content: center;
      text-align: center; padding: 24px;
    }
    #finishTrophy { font-size: 72px; }
    #finishWinner { font-size: 26px; color: #ffd166; font-weight: bold; margin: 14px 0 6px; }
    #finishYou { font-size: 18px; color: #4fc; margin-bottom: 24px; }
    #finishBack {
      padding: 12px 28px; font-size: 16px; font-family: monospace;
      background: #2a6; color: #fff; border: none; border-radius: 8px; cursor: pointer;
    }
    /* name prompt */
    #namePrompt { padding: 24px 20px; }
    #namePrompt h2 { color: #4fc; margin-bottom: 20px; }
    #namePrompt input {
      width: 100%; padding: 12px; font-size: 16px;
      background: #222; color: #ddd; border: 1px solid #444; border-radius: 6px;
      margin-bottom: 14px; font-family: monospace;
    }
    #namePrompt button {
      width: 100%; padding: 12px; font-size: 15px;
      background: #2a6; color: #fff; border: none; border-radius: 6px; cursor: pointer;
      font-family: monospace;
    }
  </style>
</head>
<body>
  <div id="notification"></div>
  <div id="finish">
    <div id="finishInner">
      <div id="finishTrophy">&#127942;</div>
      <div id="finishWinner"></div>
      <div id="finishYou"></div>
      <button id="finishResult" style="margin-bottom:10px;padding:12px 28px;font-size:16px;font-family:monospace;background:#258;color:#fff;border:none;border-radius:8px;cursor:pointer">Результаты</button>
      <button id="finishBack">Смотреть поле</button>
    </div>
  </div>
  <div id="namePrompt" style="display:none">
    <h2>Who are you?</h2>
    <input id="nameInput" type="text" placeholder="Your name" autocomplete="off" autocapitalize="words">
    <button onclick="setName()">Continue &rarr;</button>
  </div>
  <div id="main" style="display:none">
    <div id="panel">
      <div id="turn"></div>
      <div id="whoBox" style="display:flex;align-items:center;gap:10px;margin:6px 0">
        <img id="myChip" style="width:54px;height:54px;object-fit:contain;background:#000;border-radius:6px;display:none">
        <div id="who"></div>
      </div>
      <div id="route" class="label" style="margin:2px 0 6px"></div>
      <div id="players"></div>
      <div id="myturn">
        <button id="nextBtn" class="btn-waiting">Waiting...</button>
      </div>
      <button id="toResults" style="display:none;margin-top:8px;width:100%;padding:12px;font-size:15px;font-family:monospace;background:#258;color:#fff;border:none;border-radius:8px;cursor:pointer" onclick="window.location.href='/result'">&#127942; Результаты</button>
    </div>
    <img id="feed" src="/stream" alt="Game feed">
  </div>
  <script>
    let M = {};
    fetch('/api/messages').then(r => r.json()).then(d => {
      M = d.messages || {};
      document.getElementById('nameInput').placeholder = M.join_name_placeholder || '';
      document.querySelector('#namePrompt button').textContent = M.who_continue || 'Continue →';
      document.querySelector('#namePrompt h2').textContent = M.who_prompt || 'Who are you?';
    }).catch(() => {});

    let playerName = localStorage.getItem('playerName');
    let pollInterval = null;
    let eventBaseline = -1;
    let notifTimeout = null;

    function ruleMsg(data) {
      const d = data.distance || '';
      const map = {
        'SKIP_TURN':    () => M.rule_skip_turn || 'Turn skipped!',
        'MOVE_FORWARD': () => (M.rule_move_forward || 'Forward {distance}!').replace('{distance}', d),
        'MOVE_BACK':    () => (M.rule_move_back || 'Back {distance}!').replace('{distance}', d),
        'FINISH':       () => M.rule_finish || 'Finish!',
      };
      const fn = map[data.effect];
      return fn ? fn() : (M.rule_unknown || 'Rule: {effect}').replace('{effect}', data.effect || '');
    }

    function init() {
      if (!playerName) {
        document.getElementById('namePrompt').style.display = 'block';
      } else {
        showMain();
      }
    }

    function setName() {
      const v = document.getElementById('nameInput').value.trim();
      if (!v) return;
      localStorage.setItem('playerName', v);
      playerName = v;
      document.getElementById('namePrompt').style.display = 'none';
      showMain();
    }

    function showMain() {
      document.getElementById('main').style.display = 'block';
      fetchEventBaseline();
      poll();
      if (!pollInterval) {
        pollInterval = setInterval(poll, 1000);
        setInterval(pollEvents, 2000);
      }
    }

    function fetchEventBaseline() {
      fetch('/api/game/events').then(r => r.json()).then(d => {
        eventBaseline = (d.events || []).length;
      }).catch(() => {});
    }

    function pollEvents() {
      fetch('/api/game/events').then(r => r.json()).then(d => {
        const evs = d.events || [];
        if (eventBaseline < 0) { eventBaseline = evs.length; return; }
        if (evs.length > eventBaseline) {
          evs.slice(eventBaseline).forEach(ev => {
            if (ev.type === 'RULE_TRIGGERED') showNotification(ev.data);
          });
          eventBaseline = evs.length;
        }
      }).catch(() => {});
    }

    // переход «К игре (поле)» с /result помечается хэшем — сразу показать поле, без оверлея
    let finishDismissed = (location.hash === '#field');
    document.getElementById('finishBack').onclick = () => {
      finishDismissed = true;
      document.getElementById('finish').style.display = 'none';
    };
    document.getElementById('finishResult').onclick = () => { window.location.href = '/result'; };
    function showFinish(winner) {
      const f = document.getElementById('finish');
      document.getElementById('finishWinner').textContent =
        (M.game_over || 'Game over') + ': ' + winner;
      document.getElementById('finishYou').textContent =
        (winner === playerName) ? '🎉 Вы победили!' : 'Спасибо за игру';
      f.style.display = 'flex';
    }
    function hideFinish() { document.getElementById('finish').style.display = 'none'; }

    let curChipId = null;
    function setMyChip(chipId, chipName) {
      const img = document.getElementById('myChip');
      if (chipId) {
        if (chipId !== curChipId) {
          img.src = '/api/chips/' + chipId + '/image';
          curChipId = chipId;
        }
        img.style.display = 'block';
        img.onerror = () => { img.style.display = 'none'; };
      } else {
        img.style.display = 'none';
        curChipId = null;
      }
    }

    function showNotification(data) {
      const el = document.getElementById('notification');
      el.textContent = ruleMsg(data);
      el.style.display = 'block';
      el.style.opacity = '1';
      if (notifTimeout) clearTimeout(notifTimeout);
      notifTimeout = setTimeout(() => {
        el.style.opacity = '0';
        setTimeout(() => { el.style.display = 'none'; }, 400);
      }, 5000);
    }

    const ROUTE_NAMES = {linear: 'линейный', snake: 'змейка', spiral: 'спираль'};
    function poll() {
      fetch('/api/game/state').then(r => r.json()).then(data => {
        const s = data.state || {};
        const btn = document.getElementById('nextBtn');
        // маршрут поля
        document.getElementById('route').textContent =
          s.path_type ? 'Маршрут: ' + (ROUTE_NAMES[s.path_type] || s.path_type) : '';
        // постоянный доступ к результатам, пока партия завершена (даже после закрытия оверлея)
        document.getElementById('toResults').style.display = (!s.active && s.winner) ? 'block' : 'none';

        if (!s.active) {
          // экран финиша поверх всего, если есть победитель и не закрыт игроком
          if (s.winner && !finishDismissed) showFinish(s.winner); else hideFinish();
          const msg = s.winner
            ? (M.game_over || 'Game over') + ': ' + s.winner
            : (M.game_not_started || 'Game not started');
          document.getElementById('turn').textContent = msg;
          // экран ожидания: моя фишка с картинкой + список присоединившихся
          const me = (s.queue || []).find(q => q.player === playerName);
          setMyChip(me ? me.chip_id : null, me ? me.chip_name : null);
          if (me) {
            document.getElementById('who').innerHTML =
              (M.playing_as || 'You') + ': <span class="player">' + playerName + '</span>' +
              (me.chip_name ? ' &nbsp;|&nbsp; ' + (M.chip_label || 'Chip') + ': <span class="chip">' + me.chip_name + '</span>' : '');
          } else {
            document.getElementById('who').innerHTML = '';
          }
          const roster = (s.queue || []).map(q => {
            const you = q.player === playerName ? '<span class="you">' + (M.you_label||'[you]') + '</span> ' : '';
            return '<div class="player-row">' + you +
              '<span class="player">' + q.player + '</span> ' +
              '<span class="chip">' + (q.chip_name||'') + '</span></div>';
          }).join('');
          document.getElementById('players').innerHTML = roster;
          setBtn('waiting', M.btn_game_not_started || 'Waiting...');
          return;
        }

        // игра активна — сбросить экран финиша (для следующей партии)
        finishDismissed = false;
        hideFinish();

        // Player column
        const myPlayer = (s.players || []).find(p => p.name === playerName);
        if (myPlayer) {
          document.getElementById('who').innerHTML =
            (M.playing_as || 'Playing as') + ': <span class="player">' + playerName + '</span>' +
            (myPlayer.chip_name ? ' &nbsp;|&nbsp; ' + (M.chip_label || 'Chip') + ': <span class="chip">' + myPlayer.chip_name + '</span>' : '');
          setMyChip(myPlayer.chip_id, myPlayer.chip_name);
        }
        const cellLbl = M.cell_label || 'cell';
        const youLbl  = M.you_label  || '[you]';
        const skipLbl = M.skip_label || 'skip';
        const pl = (s.players || []).map(p => {
          const you = p.name === playerName ? '<span class="you">' + youLbl + '</span> ' : '';
          return '<div class="player-row">' + you +
            '<span class="player">' + p.name + '</span> ' +
            '<span class="chip">' + (p.chip_name||'') + '</span> ' +
            '<span class="label">' + cellLbl + '</span> <span class="cell">' + p.cell + '</span>' +
            (p.skip_turns > 0 ? ' <span class="rule">(' + skipLbl + ' ' + p.skip_turns + ')</span>' : '') +
            '</div>';
        }).join('');
        document.getElementById('players').innerHTML = pl;

        // Turn status
        const cur = s.current_turn;
        const turnPfx = M.turn_prefix || 'Turn';
        let t = cur
          ? turnPfx + ': <span class="player">' + cur.player + '</span> <span class="chip">[' + cur.chip_name + ']</span>'
          : '<span class="fsm">' + (M.game_ready || 'Game ready') + '</span>';
        // инструкцию о ходе показываем только когда игрок реально двигает фишку;
        // при пропуске хода (turn_state != WAITING_MOVE) ничего не показываем
        if (s.turn_state === 'WAITING_MOVE' && s.expected_cell) {
          const wmTpl = M.waiting_move || 'Dice: {dice} → cell {cell}';
          t += ' <span class="fsm">' + wmTpl.replace('{dice}', s.dice_roll).replace('{cell}', s.expected_cell) + '</span>';
        } else if (s.turn_state === 'AWAITING_RULE' && s.rule_target) {
          const arTpl = M.awaiting_rule || 'Rule → cell {cell}';
          t += ' <span class="rule">' + arTpl.replace('{cell}', s.rule_target) + '</span>';
        }
        document.getElementById('turn').innerHTML = t;

        // Button state
        const myTurn = cur && cur.player === playerName;
        if (!myTurn) {
          setBtn('waiting', M.btn_waiting || 'Waiting for your turn...');
        } else if (!s.ready_for_next) {
          const target = (s.turn_state === 'AWAITING_RULE') ? s.rule_target : s.expected_cell;
          const moveTpl = M.btn_move_chip || 'Place chip on cell {cell}';
          setBtn('moving', target ? moveTpl.replace('{cell}', target) : moveTpl.replace(' on cell {cell}', ''));
        } else {
          setBtn('ready', M.btn_next || '▶ Next Turn');
        }
      }).catch(() => {});
    }

    function setBtn(state, text) {
      const btn = document.getElementById('nextBtn');
      btn.className = 'btn-' + state;
      btn.disabled = (state !== 'ready');
      btn.textContent = text;
    }

    function next() {
      fetch('/api/game/next', { method: 'POST' }).catch(() => {});
    }

    document.getElementById('nextBtn').onclick = next;
    document.getElementById('nameInput') && document.getElementById('nameInput').addEventListener('keydown', e => {
      if (e.key === 'Enter') setName();
    });

    init();
  </script>
</body>
</html>
""".encode("utf-8")


RESULT_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Результаты</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: #0c0c0c; color: #ddd; padding: 28px 20px; text-align: center; }
    #trophy { font-size: 64px; }
    #winner { font-size: 26px; color: #ffd166; font-weight: bold; margin: 12px 0 4px; }
    #you { font-size: 17px; color: #4fc; margin-bottom: 22px; }
    #standings { max-width: 420px; margin: 0 auto 24px; text-align: left; }
    .prow {
      display: flex; align-items: center; gap: 10px; padding: 10px 12px;
      border: 1px solid #333; border-radius: 8px; margin-bottom: 8px; background: #161616;
    }
    .prow.win { border-color: #ffd166; background: #1a1606; }
    .prow.me { box-shadow: 0 0 0 1px #4fc inset; }
    .rank { font-size: 18px; color: #888; width: 28px; }
    .prow img { width: 40px; height: 40px; object-fit: contain; background: #000; border-radius: 6px; }
    .pname { color: #4fc; font-weight: bold; }
    .pchip { color: #ffd166; font-size: 13px; }
    .pcell { margin-left: auto; color: #fc4; }
    .meTag { color: #ff9; font-size: 12px; }
    #back {
      padding: 12px 28px; font-size: 16px; font-family: monospace;
      background: #2a6; color: #fff; border: none; border-radius: 8px; cursor: pointer;
    }
    #empty { color: #777; font-size: 15px; margin: 30px 0; }
  </style>
</head>
<body>
  <div id="trophy">&#127942;</div>
  <div id="winner"></div>
  <div id="you"></div>
  <div id="standings"></div>
  <div id="empty" style="display:none">Результатов пока нет — игра не завершена.</div>
  <div id="resultBtns" style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap">
    <button id="finishScreen" style="display:none;padding:12px 24px;font-size:16px;font-family:monospace;background:#258;color:#fff;border:none;border-radius:8px;cursor:pointer" onclick="window.location.href='/game'">&#127942; Экран завершения</button>
    <button id="back" onclick="window.location.href='/game#field'">&larr; К игре (поле)</button>
  </div>
  <script>
    let M = {};
    const playerName = localStorage.getItem('playerName');
    fetch('/api/messages').then(r => r.json()).then(d => { M = d.messages || {}; }).catch(() => {});

    function render() {
      fetch('/api/game/state').then(r => r.json()).then(data => {
        const s = data.state || {};
        // страница активна, пока есть завершённое состояние (есть победитель и игра не идёт)
        if (s.active) { window.location.href = '/game'; return; }
        if (!s.winner) {
          document.getElementById('winner').textContent = '';
          document.getElementById('you').textContent = '';
          document.getElementById('standings').innerHTML = '';
          document.getElementById('empty').style.display = 'block';
          document.getElementById('trophy').style.display = 'none';
          document.getElementById('finishScreen').style.display = 'none';
          return;
        }
        document.getElementById('empty').style.display = 'none';
        document.getElementById('trophy').style.display = '';
        document.getElementById('finishScreen').style.display = '';
        document.getElementById('winner').textContent = (M.game_over || 'Победитель') + ': ' + s.winner;
        document.getElementById('you').textContent =
          (s.winner === playerName) ? '🎉 Вы победили!' : (playerName ? 'Вы: ' + playerName : '');
        // финальная таблица — по клетке (дальше прошёл = выше)
        const players = (s.players || []).slice().sort((a, b) => (b.cell||0) - (a.cell||0));
        document.getElementById('standings').innerHTML = players.map((p, i) => {
          const win = p.name === s.winner ? ' win' : '';
          const me = p.name === playerName ? ' me' : '';
          const meTag = p.name === playerName ? ' <span class="meTag">[вы]</span>' : '';
          return '<div class="prow' + win + me + '">' +
            '<span class="rank">' + (i + 1) + '</span>' +
            '<img src="/api/chips/' + p.chip_id + '/image" onerror="this.style.visibility=\\'hidden\\'">' +
            '<span class="pname">' + p.name + '</span>' + meTag +
            ' <span class="pchip">' + (p.chip_name || '') + '</span>' +
            '<span class="pcell">' + (M.cell_label || 'клетка') + ' ' + p.cell + '</span></div>';
        }).join('');
      }).catch(() => {});
    }
    render();
    setInterval(render, 1500);
  </script>
</body>
</html>
""".encode("utf-8")
