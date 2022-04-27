#!/usr/bin/env python3
"""
Very simple HTTP server in python for logging requests
Usage::
    ./server.py [<port>]
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import re
import uuid

from samples.clevr.Reasoner import Reasoner


def sanitize_filename(filename: str) -> str:
    """
    Replaces all forbidden chars with '' and removes unnecessary whitespaces
    If, after sanitization, the given filename is empty, the function will return 'file_[UUID][ext]'
    :param filename: filename to be sanitized
    :return: sanitized filename
    """
    chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']

    filename = filename.translate({ord(x): '' for x in chars}).strip()
    name = re.sub(r'\.[^.]+$', '', filename)
    extension = re.search(r'(\.[^.]+$)', filename)
    extension = extension.group(1) if extension else ''

    return filename if name else f'file_{uuid.uuid4().hex}{extension}'

class S(BaseHTTPRequestHandler):
    def _set_response(self):
        print("1")
        self.send_response(200)
        print("2")
        self.send_header('Content-type', 'text/html')
        print("3")
        self.end_headers()

    def do_GET(self):
        print("4")
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        print("5")
        self._set_response()
        print("6")
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):
        result, message = self.handle_upload()

        reasoner = Reasoner()
        centers = reasoner.reason()

        print("Centers: {}".format(centers))

        print("7")
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        print("8")
        #post_data = self.rfile.read(content_length) # <--- Gets the data itself
        print("9")
        #logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
        #        str(self.path), str(self.headers), post_data.decode('utf-8'))

        #print(post_data);
        print("10")
        self._set_response()
        print("11")
        self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))

    def handle_upload(self):
        """Handle the file upload."""

        # extract boundary from headers
        boundary = re.search(f'boundary=([^;]+)', self.headers['content-type']).group(1)

        # read all bytes (headers included)
        # 'readlines()' hangs the script because it needs the EOF character to stop,
        # even if you specify how many bytes to read
        # 'file.read(nbytes).splitlines(True)' does the trick because 'read()' reads 'nbytes' bytes
        # and 'splitlines(True)' splits the file into lines and retains the newline character
        data = self.rfile.read(int(self.headers['content-length'])).splitlines(True)

        # find all filenames
        filenames = re.findall(f'{boundary}.+?filename="(.+?)"', str(data))

        if not filenames:
            return False, 'couldn\'t find file name(s).'

        filenames = [sanitize_filename(filename) for filename in filenames]

        # find all boundary occurrences in data
        boundary_indices = list((i for i, line in enumerate(data) if re.search(boundary, str(line))))

        # save file(s)
        for i in range(len(filenames)):
            # remove file headers
            file_data = data[(boundary_indices[i] + 4):boundary_indices[i + 1]]

            # join list of bytes into bytestring
            file_data = b''.join(file_data)

            # write to file
            try:
                with open(f'{filenames[i]}', 'wb') as file:
                    file.write(file_data)
            except IOError:
                return False, f'couldn\'t save {filenames[i]}.'

        return True, filenames


def run(server_class=HTTPServer, handler_class=S, port=8080):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')


if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
