// Bind the noVNC connection to this exact browser origin. Without an explicit
// host, noVNC resolves a relative path against vnc.html and repeats the prefix.
// With a host, it supplies the leading slash itself. Cover both with one URL.
function desktopUrl(href) {
    const page = new URL(href);
    if (!['http:', 'https:'].includes(page.protocol) ||
        !page.pathname.endsWith('/radar-launch.html')) {
        throw new Error('Open the GSAS-II desktop through its NDIP session link.');
    }
    const prefix = page.pathname.slice(0, -'/radar-launch.html'.length);
    const target = new URL(page.origin);
    target.pathname = prefix + '/vnc.html';
    const values = {
        autoconnect: '1', resize: 'scale', reconnect: '1', quality: '6', compression: '1',
        host: page.hostname, port: page.port || (page.protocol === 'https:' ? '443' : '80'),
        encrypt: page.protocol === 'https:' ? '1' : '0',
        path: (prefix + '/websockify').replace(/^\/+/, ''),
    };
    for (const [name, value] of Object.entries(values)) target.searchParams.set(name, value);
    return target.href;
}
window.location.replace(desktopUrl(window.location.href));
