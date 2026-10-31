#!/bin/sh
# Local test CA and a dual-purpose client/server certificate. Use distinct
# workload certificates from your organization's PKI for a real deployment.
set -eu
umask 077
target="${1:-certs}"
mkdir -p "$target"
if test -f "$target/ca.key"; then
  echo "Certificate directory already initialized; refusing to replace it."
  exit 1
fi
cat > "$target/ca.cnf" <<'EOF'
[req]
distinguished_name=dn
x509_extensions=v3_ca
prompt=no
[dn]
CN=Stageplane-Local-Test-CA
[v3_ca]
basicConstraints=critical,CA:TRUE
keyUsage=critical,keyCertSign,cRLSign
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid:always,issuer
EOF
openssl req -x509 -newkey rsa:2048 -nodes -keyout "$target/ca.key" -out "$target/ca.crt" -days 2 -config "$target/ca.cnf" >/dev/null 2>&1
openssl req -newkey rsa:2048 -nodes -keyout "$target/worker.key" -out "$target/worker.csr" -subj /CN=localhost >/dev/null 2>&1
cat > "$target/extensions.cnf" <<'EOF'
subjectAltName=DNS:localhost,IP:127.0.0.1
extendedKeyUsage=serverAuth,clientAuth
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid,issuer
EOF
openssl x509 -req -in "$target/worker.csr" -CA "$target/ca.crt" -CAkey "$target/ca.key" -CAcreateserial -out "$target/worker.crt" -days 2 -extfile "$target/extensions.cnf" >/dev/null 2>&1
echo "Local test certificates created in $target"
