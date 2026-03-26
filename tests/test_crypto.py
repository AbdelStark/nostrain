from __future__ import annotations

import unittest

from nostrain.crypto import schnorr_sign, schnorr_verify, secret_key_to_public_key


class SchnorrCryptoTests(unittest.TestCase):
    def test_bip340_vector_zero_matches_official_reference(self) -> None:
        secret_key = "0000000000000000000000000000000000000000000000000000000000000003"
        public_key = "f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9"
        aux_rand = bytes.fromhex("00" * 32)
        message = bytes.fromhex("00" * 32)
        signature = (
            "e907831f80848d1069a5371b402410364bdf1c5f8307b0084c55f1ce2dca8215"
            "25f66a4a85ea8b71e482a74f382d2ce5ebeee8fdb2172f477df4900d310536c0"
        )

        self.assertEqual(secret_key_to_public_key(secret_key), public_key)
        self.assertEqual(schnorr_sign(message, secret_key, aux_rand=aux_rand), signature)
        self.assertTrue(schnorr_verify(message, public_key, signature))

    def test_bip340_vector_one_matches_official_reference(self) -> None:
        secret_key = "b7e151628aed2a6abf7158809cf4f3c762e7160f38b4da56a784d9045190cfef"
        public_key = "dff1d77f2a671c5f36183726db2341be58feae1da2deced843240f7b502ba659"
        aux_rand = bytes.fromhex("00" * 31 + "01")
        message = bytes.fromhex("243f6a8885a308d313198a2e03707344a4093822299f31d0082efa98ec4e6c89")
        signature = (
            "6896bd60eeae296db48a229ff71dfe071bde413e6d43f917dc8dcf8c78de3341"
            "8906d11ac976abccb20b091292bff4ea897efcb639ea871cfa95f6de339e4b0a"
        )

        self.assertEqual(secret_key_to_public_key(secret_key), public_key)
        self.assertEqual(schnorr_sign(message, secret_key, aux_rand=aux_rand), signature)
        self.assertTrue(schnorr_verify(message, public_key, signature))

    def test_verify_rejects_invalid_signature_vector(self) -> None:
        public_key = "dff1d77f2a671c5f36183726db2341be58feae1da2deced843240f7b502ba659"
        message = bytes.fromhex("243f6a8885a308d313198a2e03707344a4093822299f31d0082efa98ec4e6c89")
        invalid_signature = (
            "fff97bd5755eeea420453a14355235d382f6472f8568a18b2f057a1460297556"
            "3cc27944640ac607cd107ae10923d9ef7a73c643e166be5ebeafa34b1ac553e2"
        )

        self.assertFalse(schnorr_verify(message, public_key, invalid_signature))


if __name__ == "__main__":
    unittest.main()
