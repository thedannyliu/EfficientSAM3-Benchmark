from __future__ import annotations

import unittest

from sam_backend.thor_resources import parse_meminfo, parse_tegrastats


class ThorResourcesTest(unittest.TestCase):
    def test_parse_meminfo_converts_kib_to_bytes(self) -> None:
        parsed = parse_meminfo(
            "MemTotal:       1024 kB\n"
            "MemAvailable:    512 kB\n"
            "SwapTotal:       256 kB\n"
            "SwapFree:        128 kB\n"
            "Buffers:          64 kB\n"
        )

        self.assertEqual(
            parsed,
            {
                "memtotal_bytes": 1024 * 1024,
                "memavailable_bytes": 512 * 1024,
                "swaptotal_bytes": 256 * 1024,
                "swapfree_bytes": 128 * 1024,
            },
        )

    def test_parse_thor_tegrastats_sample(self) -> None:
        parsed = parse_tegrastats(
            "08-07-2026 17:50:34 RAM 42435/125772MB (lfb 5x4MB) "
            "CPU [7%@972] cpu@52.906C tj@56.906C gpu@56.5C "
            "VDD_GPU 36196mW/36393mW VIN 68862mW/56658mW"
        )

        self.assertEqual(parsed["ram_used_mb"], 42435)
        self.assertEqual(parsed["ram_total_mb"], 125772)
        self.assertEqual(parsed["temperatures_c"]["gpu"], 56.5)
        self.assertEqual(parsed["power_mw"]["VDD_GPU"], {"current": 36196, "average": 36393})
        self.assertEqual(parsed["power_mw"]["VIN"], {"current": 68862, "average": 56658})

    def test_parse_tegrastats_discrete_style_utilization(self) -> None:
        parsed = parse_tegrastats("RAM 4388/62801MB GR3D_FREQ 42%@306 EMC_FREQ 3%@2133")

        self.assertEqual(parsed["gr3d_percent"], 42.0)
        self.assertEqual(parsed["emc_percent"], 3.0)


if __name__ == "__main__":
    unittest.main()
