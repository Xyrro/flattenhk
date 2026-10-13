# Vendored third-party assets

`leaflet-1.9.4.min.js` and `leaflet-1.9.4.css` are Leaflet 1.9.4, obtained
upstream from the npm registry
(`https://registry.npmjs.org/leaflet/-/leaflet-1.9.4.tgz`, retrieved
2026-09-16; byte-identical here to upstream's copies) and redistributed
unmodified under the BSD 2-Clause licence (Copyright (c) 2010-2023
Volodymyr Agafonkin, (c) 2010-2011 CloudMade). The licence text is
`LICENSE-leaflet.txt`, from the same tarball; the build copies it into
`site/` beside the hashed Leaflet files.

They are inlined into `outputs/hk_flat_routes_map.html` and
`outputs/hk_flat_route_finder.html` so that those pages are genuinely
self-contained files: each opens and works with no CDN and no network
access (the route finder asks Google Fonts for its typeface and falls back
to system fonts without it).

## The certificate the iGeoCom host leaves out

`hongkongpost-ecert-ssl-ca-3-17.pem` is not code. It is the public
certificate of an intermediate certificate authority, *Hongkong Post e-Cert
SSL CA 3 - 17*, issued by *Hongkong Post Root CA 3*. The Lands Department's
open-data host (`open.hkmapservice.gov.hk`, where the iGeoCom file is
fetched) presents its own certificate and that root but not this
certificate between them, so Python, which does not go and fetch a missing
link the way a browser does, cannot verify the host. `download.py` adds
this one certificate to the roots it already trusts, for that one request,
instead of skipping the check or depending on the system's `curl`.

- Retrieved 2026-10-13 from
  `http://www1.eCert.gov.hk/root/ecert_ssl_ca_3-17.crt`, the address the
  host's own certificate gives for its issuer (Authority Information
  Access), and converted from DER to PEM unchanged.
- Serial `3DB180EF6FF85F7A895D16445B9FC9FC3DEFBBC7`, valid from 20 March
  2025 to 3 June 2032.
- SHA-256 fingerprint of the certificate:
  `CB:73:0C:3A:5F:E9:58:B8:BD:C5:B3:63:89:2C:9F:53:0B:33:F9:8A:43:CE:3C:25:78:FC:E0:96:B4:AE:23:3C`.
- Checked the same day with `openssl verify`: it chains to the Hongkong
  Post Root CA 3 in certifi 2026.07.22, the host's certificate verifies
  with it, and fails ("unable to get local issuer certificate") without
  it. Trusting it therefore adds nothing that the root did not already
  cover.

If the host moves to another intermediate, or starts sending its whole
chain, this file stops mattering: the request is first tried against the
ordinary roots, and falls back to `curl` when this certificate does not
complete the chain either.
