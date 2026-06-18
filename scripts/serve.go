package main

import (
	"embed"
	"flag"
	"io/fs"
	"log"
	"mime"
	"net"
	"net/http"
	"os"
	"path"
	"strings"
)

//go:embed all:dist
var dist embed.FS

func main() {
	port := flag.String("port", "", "listen port (default 8080, or $PORT)")
	flag.Parse()

	if *port == "" {
		*port = os.Getenv("PORT")
	}
	if *port == "" {
		*port = "8080"
	}

	sub, err := fs.Sub(dist, "dist")
	if err != nil {
		log.Fatalf("embedded dist not found: %v", err)
	}

	ln, err := net.Listen("tcp", ":"+*port)
	if err != nil {
		log.Fatalf("listen :%s: %v", *port, err)
	}

	log.Printf("serving at http://localhost:%s", *port)
	if err := http.Serve(ln, spaHandler{fs: sub}); err != nil {
		log.Fatalf("serve: %v", err)
	}
}

type spaHandler struct {
	fs fs.FS
}

func (h spaHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	p := strings.TrimPrefix(r.URL.Path, "/")
	if p == "" {
		p = "index.html"
	}

	data, err := fs.ReadFile(h.fs, p)
	if err != nil {
		data, err = fs.ReadFile(h.fs, "index.html")
		if err != nil {
			http.NotFound(w, r)
			return
		}
	}

	ctype := mime.TypeByExtension(path.Ext(p))
	if ctype != "" {
		w.Header().Set("Content-Type", ctype)
	}

	w.Write(data)
}
