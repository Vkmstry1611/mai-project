"""Create a private, local-only TLS certificate for phone microphone access."""
from datetime import datetime, timedelta, timezone
from ipaddress import ip_address
from pathlib import Path


def local_certificates(directory: Path, lan_ip: str):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    ip = ip_address(lan_ip)
    if not ip.is_private:
        raise ValueError('The local HTTPS address must be a private LAN IP.')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / str(ip).replace(':', '_')
    ca_file, cert_file, key_file = (prefix.with_name(prefix.name + suffix)
                                    for suffix in ('-ca.pem', '-server.pem', '-server-key.pem'))
    phone_file = prefix.with_name(prefix.name + '-phone-ca.crt')
    if all(path.is_file() for path in (ca_file, cert_file, key_file)):
        current = x509.load_pem_x509_certificate(cert_file.read_bytes())
        if current.not_valid_after_utc > datetime.now(timezone.utc) + timedelta(days=1):
            if not phone_file.is_file():
                ca = x509.load_pem_x509_certificate(ca_file.read_bytes())
                phone_file.write_bytes(ca.public_bytes(serialization.Encoding.DER))
            return ca_file, cert_file, key_file

    now = datetime.now(timezone.utc)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'VisionBridge local phone CA')])
    ca = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name)
          .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
          .not_valid_before(now - timedelta(minutes=5)).not_valid_after(now + timedelta(days=30))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(digital_signature=True, key_encipherment=False,
              key_cert_sign=True, key_agreement=False, content_commitment=False,
              data_encipherment=False, crl_sign=True, encipher_only=False,
              decipher_only=False), critical=True)
          .sign(ca_key, hashes.SHA256()))
    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'VisionBridge local app')])
    server = (x509.CertificateBuilder().subject_name(server_name).issuer_name(ca_name)
              .public_key(server_key.public_key()).serial_number(x509.random_serial_number())
              .not_valid_before(now - timedelta(minutes=5)).not_valid_after(now + timedelta(days=30))
              .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
              .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ip),
                  x509.IPAddress(ip_address('127.0.0.1')), x509.DNSName('localhost')]), critical=False)
              .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
              .sign(ca_key, hashes.SHA256()))
    ca_file.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    phone_file.write_bytes(ca.public_bytes(serialization.Encoding.DER))
    cert_file.write_bytes(server.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(server_key.private_bytes(serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
    return ca_file, cert_file, key_file
